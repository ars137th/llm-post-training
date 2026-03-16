"""
PPO Trainer with Custom Training Loop

Implements the complete PPO algorithm for RLHF:
1. Rollout phase: Generate responses, collect rewards
2. Update phase: Optimize policy and value function

Unlike DPO (which uses HuggingFace Trainer), PPO requires a custom loop
because of its two-phase structure and on-policy nature.

The trainer manages four models:
- Actor (policy): Model being optimized
- Critic (value function): Estimates expected rewards
- Reference: Frozen SFT model (KL constraint)
- Reward Model: Scores responses (frozen)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
import time

from .loss import (
    compute_log_probs,
    ppo_loss,
    value_loss,
    kl_divergence,
    compute_rlhf_reward,
    policy_entropy_loss,
    total_ppo_loss,
    check_ppo_ratio,
)
from .buffer import RolloutBuffer, RolloutBatch
from ...models.reward import RewardModel


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Rollout
    batch_size: int = 32  # Number of prompts per rollout
    max_prompt_length: int = 512
    max_response_length: int = 256

    # Generation
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # PPO hyperparameters
    ppo_epochs: int = 4  # Update epochs per rollout
    mini_batch_size: int = 8  # Mini-batch size for updates
    clip_range: float = 0.2  # PPO clip parameter (epsilon)
    clip_range_vf: Optional[float] = None  # Value function clipping (optional)

    # GAE
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda

    # Loss coefficients
    vf_coef: float = 0.5  # Value loss coefficient
    ent_coef: float = 0.01  # Entropy coefficient
    kl_coef: float = 0.05  # KL penalty coefficient (beta)

    # Optimization
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training
    num_rollouts: int = 100  # Total number of rollout iterations
    log_every: int = 1  # Log every N rollouts
    save_every: int = 10  # Save checkpoint every N rollouts

    # Adaptive KL (optional)
    use_adaptive_kl: bool = False
    target_kl: float = 0.01  # Target KL per token
    kl_horizon: int = 10000  # Horizon for KL controller

    # Normalization
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    reward_clip: Optional[float] = None  # Clip rewards to [-X, X]

    # Device
    device: str = "auto"


class PPOTrainer:
    """
    Custom trainer for PPO with rollout-update loop.

    Training loop:
        for iteration in range(num_rollouts):
            # Phase 1: Rollout (generate data)
            rollout_buffer = rollout(prompts)

            # Phase 2: Update (optimize policy)
            for epoch in range(ppo_epochs):
                for batch in rollout_buffer.sample_batches():
                    loss = compute_loss(batch)
                    loss.backward()
                    optimizer.step()

            log_metrics()
    """

    def __init__(
        self,
        actor: PreTrainedModel,
        critic: RewardModel,  # RewardModel with value head
        reference: PreTrainedModel,
        reward_model: RewardModel,
        tokenizer: PreTrainedTokenizer,
        config: PPOConfig,
        optimizer: Optional[Optimizer] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            actor: Policy model (trainable)
            critic: Value function (RewardModel with value head, trainable)
            reference: Reference policy (frozen)
            reward_model: Reward model (frozen)
            tokenizer: Tokenizer
            config: PPO configuration
            optimizer: Custom optimizer (optional, will create AdamW if None)
        """
        self.actor = actor
        self.critic = critic
        self.reference = reference
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Move models to device
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.reference = self.reference.to(self.device)
        self.reward_model = self.reward_model.to(self.device)

        # Freeze reference and reward models
        self.reference.eval()
        self.reward_model.eval()
        for param in self.reference.parameters():
            param.requires_grad = False
        for param in self.reward_model.model.parameters():
            param.requires_grad = False

        # Setup optimizer
        if optimizer is None:
            # Optimize both actor and critic
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            self.optimizer = AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            gamma=config.gamma,
            lam=config.lam,
            normalize_advantages=config.normalize_advantages,
        )

        # Training metrics
        self.training_metrics = {
            'rollout': [],
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'reward': [],
            'kl': [],
            'clip_fraction': [],
            'approx_kl': [],
            'explained_variance': [],
        }

        # Adaptive KL controller
        if config.use_adaptive_kl:
            self.kl_ctl = AdaptiveKLController(config.kl_coef, config.target_kl)
        else:
            self.kl_ctl = FixedKLController(config.kl_coef)

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate responses using the actor (policy).

        Args:
            prompts: List of prompt strings

        Returns:
            responses: List of generated response strings
            response_ids: Generated token IDs [batch_size, response_len]
            log_probs: Log probabilities [batch_size]
        """
        self.actor.eval()

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            max_length=self.config.max_prompt_length,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        # Generate responses
        outputs = self.actor.generate(
            input_ids=prompt_encodings['input_ids'],
            attention_mask=prompt_encodings['attention_mask'],
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Extract generated sequences (prompt + response)
        generated_ids = outputs.sequences  # [batch_size, prompt_len + response_len]

        # Get response portion only
        prompt_len = prompt_encodings['input_ids'].shape[1]
        response_ids = generated_ids[:, prompt_len:]

        # Decode responses
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Compute log probabilities for generated sequences
        # Forward pass to get logits
        forward_outputs = self.actor(
            input_ids=generated_ids,
            attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
        )

        log_probs = compute_log_probs(
            logits=forward_outputs.logits,
            labels=generated_ids,
            attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
        )

        self.actor.train()

        return responses, response_ids, log_probs

    @torch.no_grad()
    def compute_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities from reference model.

        Args:
            input_ids: Full sequences (prompt + response) [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            log_probs: Log probabilities [batch_size]
        """
        outputs = self.reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        log_probs = compute_log_probs(
            logits=outputs.logits,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        return log_probs

    @torch.no_grad()
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        Compute rewards using reward model.

        Args:
            prompts: Prompt strings
            responses: Response strings

        Returns:
            rewards: Reward model scores [batch_size]
        """
        # Combine prompts and responses
        full_texts = [p + r for p, r in zip(prompts, responses)]

        # Tokenize
        encodings = self.tokenizer(
            full_texts,
            padding=True,
            max_length=self.config.max_prompt_length + self.config.max_response_length,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        # Get rewards
        rewards = self.reward_model(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            return_dict=False,
        )

        # Clip rewards if specified
        if self.config.reward_clip is not None:
            rewards = torch.clamp(rewards, -self.config.reward_clip, self.config.reward_clip)

        return rewards

    @torch.no_grad()
    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value estimates using critic.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            values: Value estimates [batch_size]
        """
        # Forward through critic (RewardModel with value head)
        # RewardModel.forward returns scalar values directly
        values = self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,  # Get tensor directly
        )

        # Ensure it's 1D
        if values.dim() > 1:
            values = values.squeeze()

        return values

    def rollout(self, prompts: List[str]) -> RolloutBuffer:
        """
        Perform rollout phase: generate responses and collect data.

        Args:
            prompts: List of prompt strings

        Returns:
            RolloutBuffer with collected trajectories
        """
        # Clear buffer
        self.rollout_buffer.clear()

        # Process in batches
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i:i + self.config.batch_size]

            # Generate responses
            responses, response_ids, old_log_probs = self.generate_responses(batch_prompts)

            # Tokenize full sequences
            full_texts = [p + r for p, r in zip(batch_prompts, responses)]
            full_encodings = self.tokenizer(
                full_texts,
                padding=True,
                max_length=self.config.max_prompt_length + self.config.max_response_length,
                truncation=True,
                return_tensors='pt',
            ).to(self.device)

            input_ids = full_encodings['input_ids']
            attention_mask = full_encodings['attention_mask']

            # Compute reference log probs (for KL penalty)
            ref_log_probs = self.compute_ref_log_probs(input_ids, attention_mask)

            # Compute rewards
            rm_rewards = self.compute_rewards(batch_prompts, responses)

            # Compute total rewards (RM reward - KL penalty)
            rewards = compute_rlhf_reward(
                reward_model_scores=rm_rewards,
                log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                kl_coef=self.kl_ctl.value,
            )

            # Compute values
            values = self.compute_values(input_ids, attention_mask)

            # Add to buffer
            prompt_encodings = self.tokenizer(
                batch_prompts,
                padding=True,
                max_length=self.config.max_prompt_length,
                truncation=True,
                return_tensors='pt',
            )

            response_encodings = self.tokenizer(
                responses,
                padding=True,
                max_length=self.config.max_response_length,
                truncation=True,
                return_tensors='pt',
            )

            self.rollout_buffer.add(
                prompt_input_ids=prompt_encodings['input_ids'],
                prompt_attention_mask=prompt_encodings['attention_mask'],
                response_input_ids=response_encodings['input_ids'],
                response_attention_mask=response_encodings['attention_mask'],
                input_ids=input_ids,
                attention_mask=attention_mask,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                rewards=rewards,
                values=values,
            )

        # Compute advantages
        self.rollout_buffer.compute_advantages()

        return self.rollout_buffer

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform update phase: optimize policy and value function.

        Args:
            rollout_buffer: Buffer with rollout data

        Returns:
            Dict with training metrics
        """
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'approx_kl': [],
            'explained_variance': [],
        }

        # Multiple epochs over same data
        for epoch in range(self.config.ppo_epochs):
            # Sample mini-batches
            for batch in rollout_buffer.sample_batches(
                batch_size=self.config.mini_batch_size,
                device=self.device,
                num_epochs=1,
                shuffle=True,
            ):
                # Forward through actor
                actor_outputs = self.actor(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )

                # Compute new log probs
                new_log_probs = compute_log_probs(
                    logits=actor_outputs.logits,
                    labels=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )

                # Compute PPO loss
                p_loss, p_details = ppo_loss(
                    log_probs=new_log_probs,
                    old_log_probs=batch.old_log_probs,
                    advantages=batch.advantages,
                    clip_range=self.config.clip_range,
                    return_details=True,
                )

                # Compute new values
                new_values = self.compute_values(batch.input_ids, batch.attention_mask)

                # Compute value loss
                v_loss, v_details = value_loss(
                    values=new_values,
                    returns=batch.returns,
                    old_values=batch.values,
                    clip_range_vf=self.config.clip_range_vf,
                    return_details=True,
                )

                # Compute entropy loss
                e_loss, e_details = policy_entropy_loss(
                    logits=actor_outputs.logits,
                    attention_mask=batch.attention_mask,
                    return_details=True,
                )

                # Total loss
                loss = total_ppo_loss(
                    policy_loss=p_loss,
                    value_loss=v_loss,
                    entropy_loss=e_loss,
                    vf_coef=self.config.vf_coef,
                    ent_coef=self.config.ent_coef,
                )

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.config.max_grad_norm
                    )

                self.optimizer.step()

                # Record metrics
                metrics['policy_loss'].append(p_details['loss'])
                metrics['value_loss'].append(v_details['loss'])
                metrics['entropy_loss'].append(e_details['entropy_loss'])
                metrics['total_loss'].append(loss.item())
                metrics['clip_fraction'].append(p_details['clip_fraction'])
                metrics['approx_kl'].append(p_details['approx_kl'])
                metrics['explained_variance'].append(v_details['explained_variance'])

        # Average metrics
        averaged_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

        # Update KL controller
        mean_kl = averaged_metrics['approx_kl']
        self.kl_ctl.update(mean_kl, len(rollout_buffer))

        return averaged_metrics

    def train(self, prompt_dataset: List[str]) -> Dict[str, List]:
        """
        Main training loop.

        Args:
            prompt_dataset: List of prompt strings

        Returns:
            Dict with training history
        """
        print(f"Starting PPO training on {len(prompt_dataset)} prompts")
        print(f"Device: {self.device}")
        print(f"Rollouts: {self.config.num_rollouts}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"PPO epochs: {self.config.ppo_epochs}")

        for rollout_idx in range(self.config.num_rollouts):
            start_time = time.time()

            # Sample prompts for this rollout
            # In practice, you might want to iterate through the dataset
            # For simplicity, we randomly sample here
            import random
            sampled_prompts = random.sample(
                prompt_dataset,
                min(self.config.batch_size, len(prompt_dataset))
            )

            # Phase 1: Rollout
            rollout_buffer = self.rollout(sampled_prompts)

            # Phase 2: Update
            update_metrics = self.update(rollout_buffer)

            # Get rollout stats
            buffer_stats = rollout_buffer.get_stats()

            # Log metrics
            self.training_metrics['rollout'].append(rollout_idx)
            self.training_metrics['loss'].append(update_metrics['total_loss'])
            self.training_metrics['policy_loss'].append(update_metrics['policy_loss'])
            self.training_metrics['value_loss'].append(update_metrics['value_loss'])
            self.training_metrics['reward'].append(buffer_stats.get('reward_mean', 0))
            self.training_metrics['kl'].append(update_metrics['approx_kl'])
            self.training_metrics['clip_fraction'].append(update_metrics['clip_fraction'])
            self.training_metrics['explained_variance'].append(update_metrics['explained_variance'])

            # Print progress
            if (rollout_idx + 1) % self.config.log_every == 0:
                elapsed = time.time() - start_time
                print(f"\n[Rollout {rollout_idx + 1}/{self.config.num_rollouts}]")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Reward: {buffer_stats.get('reward_mean', 0):.4f} ± {buffer_stats.get('reward_std', 0):.4f}")
                print(f"  Policy Loss: {update_metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {update_metrics['value_loss']:.4f}")
                print(f"  KL: {update_metrics['approx_kl']:.6f}")
                print(f"  Clip Fraction: {update_metrics['clip_fraction']:.2%}")
                print(f"  Explained Variance: {update_metrics['explained_variance']:.4f}")
                print(f"  KL Coef: {self.kl_ctl.value:.6f}")

        print("\n✅ PPO training complete!")
        return self.training_metrics

    def save(self, path: str):
        """Save actor and critic models."""
        actor_path = f"{path}/actor"
        critic_path = f"{path}/critic"

        self.actor.save_pretrained(actor_path)
        self.critic.save_pretrained(critic_path)

        print(f"Saved models to {path}/")


# KL Controllers

class FixedKLController:
    """Fixed KL coefficient."""
    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, kl: float, n_steps: int):
        """No update for fixed controller."""
        pass


class AdaptiveKLController:
    """Adaptive KL coefficient based on target KL."""
    def __init__(self, init_kl_coef: float, target: float, horizon: int = 10000):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, kl: float, n_steps: int):
        """Update KL coefficient based on current KL."""
        proportional_error = kl - self.target
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
        self.value = max(0.0, self.value)  # Keep positive
