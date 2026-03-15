"""
Rollout Buffer for PPO

Stores trajectories collected during the rollout phase:
- Prompts and generated responses
- Log probabilities from actor (policy)
- Rewards from reward model (with KL penalty)
- Value estimates from critic
- Computed advantages (via GAE)

The buffer supports:
- Adding trajectories
- Computing advantages
- Sampling mini-batches for updates
- Clearing between rollouts
"""

from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass, field


@dataclass
class RolloutBatch:
    """A batch of rollout data for training."""
    # Input data
    prompt_input_ids: torch.Tensor  # [batch_size, prompt_len]
    prompt_attention_mask: torch.Tensor  # [batch_size, prompt_len]
    response_input_ids: torch.Tensor  # [batch_size, response_len]
    response_attention_mask: torch.Tensor  # [batch_size, response_len]

    # Full sequences (prompt + response)
    input_ids: torch.Tensor  # [batch_size, total_len]
    attention_mask: torch.Tensor  # [batch_size, total_len]

    # Collected during rollout
    old_log_probs: torch.Tensor  # [batch_size] - from actor during generation
    ref_log_probs: torch.Tensor  # [batch_size] - from reference model
    rewards: torch.Tensor  # [batch_size] - from reward model
    values: torch.Tensor  # [batch_size] - from critic

    # Computed after rollout
    advantages: torch.Tensor  # [batch_size] - from GAE
    returns: torch.Tensor  # [batch_size] - advantages + values (targets for critic)

    def to(self, device: torch.device) -> 'RolloutBatch':
        """Move all tensors to device."""
        return RolloutBatch(
            prompt_input_ids=self.prompt_input_ids.to(device),
            prompt_attention_mask=self.prompt_attention_mask.to(device),
            response_input_ids=self.response_input_ids.to(device),
            response_attention_mask=self.response_attention_mask.to(device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            ref_log_probs=self.ref_log_probs.to(device),
            rewards=self.rewards.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )

    def __len__(self) -> int:
        """Return batch size."""
        return self.input_ids.shape[0]


class RolloutBuffer:
    """
    Buffer for storing PPO rollout data.

    During rollout phase:
    1. Generate responses with actor
    2. Collect log probs, rewards, values
    3. Store in buffer

    After rollout phase:
    4. Compute advantages via GAE
    5. Sample mini-batches for updates

    During update phase:
    6. Iterate through mini-batches multiple times (epochs)
    7. Update actor and critic
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_advantages: bool = True,
    ):
        """
        Initialize rollout buffer.

        Args:
            gamma: Discount factor for GAE
            lam: Lambda parameter for GAE
            normalize_advantages: Whether to normalize advantages
        """
        self.gamma = gamma
        self.lam = lam
        self.normalize_advantages = normalize_advantages

        # Storage
        self.prompt_input_ids: List[torch.Tensor] = []
        self.prompt_attention_mask: List[torch.Tensor] = []
        self.response_input_ids: List[torch.Tensor] = []
        self.response_attention_mask: List[torch.Tensor] = []
        self.input_ids: List[torch.Tensor] = []
        self.attention_mask: List[torch.Tensor] = []
        self.old_log_probs: List[torch.Tensor] = []
        self.ref_log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []

        # Computed
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def add(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Add a batch of trajectories to the buffer.

        Args:
            prompt_input_ids: Tokenized prompts [batch_size, prompt_len]
            prompt_attention_mask: Prompt attention mask [batch_size, prompt_len]
            response_input_ids: Generated responses [batch_size, response_len]
            response_attention_mask: Response attention mask [batch_size, response_len]
            input_ids: Full sequences (prompt + response) [batch_size, total_len]
            attention_mask: Full attention mask [batch_size, total_len]
            old_log_probs: Log probs from actor [batch_size]
            ref_log_probs: Log probs from reference [batch_size]
            rewards: Rewards from reward model [batch_size]
            values: Values from critic [batch_size]
        """
        self.prompt_input_ids.append(prompt_input_ids.cpu())
        self.prompt_attention_mask.append(prompt_attention_mask.cpu())
        self.response_input_ids.append(response_input_ids.cpu())
        self.response_attention_mask.append(response_attention_mask.cpu())
        self.input_ids.append(input_ids.cpu())
        self.attention_mask.append(attention_mask.cpu())
        self.old_log_probs.append(old_log_probs.cpu())
        self.ref_log_probs.append(ref_log_probs.cpu())
        self.rewards.append(rewards.cpu())
        self.values.append(values.cpu())

    def compute_advantages(self):
        """
        Compute advantages and returns using GAE.

        Must be called after all trajectories are added and before sampling.
        """
        from .gae import compute_gae, compute_value_targets, normalize_advantages

        # Concatenate all batches
        rewards = torch.cat(self.rewards, dim=0)  # [total_size]
        values = torch.cat(self.values, dim=0)    # [total_size]

        # Add terminal value (0)
        values_with_terminal = torch.cat([values, torch.zeros(1)])  # [total_size + 1]

        # Compute advantages via GAE
        advantages = compute_gae(
            rewards=rewards,
            values=values_with_terminal,
            gamma=self.gamma,
            lam=self.lam,
        )

        # Normalize advantages
        if self.normalize_advantages:
            advantages = normalize_advantages(advantages)

        # Compute returns (targets for value function)
        returns = compute_value_targets(advantages, values_with_terminal)

        self.advantages = advantages
        self.returns = returns

    def get_all(self, device: torch.device) -> RolloutBatch:
        """
        Get all data as a single batch.

        Args:
            device: Device to move data to

        Returns:
            RolloutBatch with all collected data
        """
        if self.advantages is None:
            raise ValueError("Must call compute_advantages() before getting data")

        batch = RolloutBatch(
            prompt_input_ids=torch.cat(self.prompt_input_ids, dim=0),
            prompt_attention_mask=torch.cat(self.prompt_attention_mask, dim=0),
            response_input_ids=torch.cat(self.response_input_ids, dim=0),
            response_attention_mask=torch.cat(self.response_attention_mask, dim=0),
            input_ids=torch.cat(self.input_ids, dim=0),
            attention_mask=torch.cat(self.attention_mask, dim=0),
            old_log_probs=torch.cat(self.old_log_probs, dim=0),
            ref_log_probs=torch.cat(self.ref_log_probs, dim=0),
            rewards=torch.cat(self.rewards, dim=0),
            values=torch.cat(self.values, dim=0),
            advantages=self.advantages,
            returns=self.returns,
        )

        return batch.to(device)

    def sample_batches(
        self,
        batch_size: int,
        device: torch.device,
        num_epochs: int = 1,
        shuffle: bool = True,
    ) -> List[RolloutBatch]:
        """
        Sample mini-batches for training.

        Args:
            batch_size: Size of each mini-batch
            device: Device to move data to
            num_epochs: Number of times to iterate through data
            shuffle: Whether to shuffle data each epoch

        Yields:
            Mini-batches of RolloutBatch
        """
        if self.advantages is None:
            raise ValueError("Must call compute_advantages() before sampling")

        total_size = len(self.input_ids[0])
        for tensor_list in [
            self.prompt_input_ids,
            self.response_input_ids,
            self.input_ids,
            self.old_log_probs,
            self.rewards,
        ]:
            total_size = sum(t.shape[0] for t in tensor_list)
            break

        # Get all data as single batch
        full_batch = self.get_all(device)

        for epoch in range(num_epochs):
            # Shuffle indices
            if shuffle:
                indices = torch.randperm(len(full_batch))
            else:
                indices = torch.arange(len(full_batch))

            # Create mini-batches
            for start_idx in range(0, len(full_batch), batch_size):
                end_idx = min(start_idx + batch_size, len(full_batch))
                batch_indices = indices[start_idx:end_idx]

                # Extract mini-batch
                mini_batch = RolloutBatch(
                    prompt_input_ids=full_batch.prompt_input_ids[batch_indices],
                    prompt_attention_mask=full_batch.prompt_attention_mask[batch_indices],
                    response_input_ids=full_batch.response_input_ids[batch_indices],
                    response_attention_mask=full_batch.response_attention_mask[batch_indices],
                    input_ids=full_batch.input_ids[batch_indices],
                    attention_mask=full_batch.attention_mask[batch_indices],
                    old_log_probs=full_batch.old_log_probs[batch_indices],
                    ref_log_probs=full_batch.ref_log_probs[batch_indices],
                    rewards=full_batch.rewards[batch_indices],
                    values=full_batch.values[batch_indices],
                    advantages=full_batch.advantages[batch_indices],
                    returns=full_batch.returns[batch_indices],
                )

                yield mini_batch

    def clear(self):
        """Clear all stored data."""
        self.prompt_input_ids.clear()
        self.prompt_attention_mask.clear()
        self.response_input_ids.clear()
        self.response_attention_mask.clear()
        self.input_ids.clear()
        self.attention_mask.clear()
        self.old_log_probs.clear()
        self.ref_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """Return number of trajectories in buffer."""
        if not self.rewards:
            return 0
        return sum(r.shape[0] for r in self.rewards)

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about stored data."""
        if not self.rewards:
            return {}

        rewards = torch.cat(self.rewards, dim=0)
        values = torch.cat(self.values, dim=0)

        stats = {
            'buffer_size': len(self),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_min': rewards.min().item(),
            'reward_max': rewards.max().item(),
            'value_mean': values.mean().item(),
            'value_std': values.std().item(),
        }

        if self.advantages is not None:
            stats.update({
                'advantage_mean': self.advantages.mean().item(),
                'advantage_std': self.advantages.std().item(),
                'advantage_min': self.advantages.min().item(),
                'advantage_max': self.advantages.max().item(),
            })

        return stats


# Utility function for creating batches from raw data

def create_rollout_batch(
    prompt_texts: List[str],
    response_texts: List[str],
    tokenizer,
    max_prompt_length: int = 512,
    max_response_length: int = 256,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, ...]:
    """
    Create tokenized batch for rollout.

    Args:
        prompt_texts: List of prompt strings
        response_texts: List of response strings
        tokenizer: Tokenizer
        max_prompt_length: Max prompt length
        max_response_length: Max response length
        device: Device to put tensors on

    Returns:
        Tuple of (prompt_input_ids, prompt_attention_mask,
                  response_input_ids, response_attention_mask,
                  input_ids, attention_mask)
    """
    # Tokenize prompts
    prompt_encodings = tokenizer(
        prompt_texts,
        padding='max_length',
        max_length=max_prompt_length,
        truncation=True,
        return_tensors='pt',
    )

    # Tokenize responses
    response_encodings = tokenizer(
        response_texts,
        padding='max_length',
        max_length=max_response_length,
        truncation=True,
        return_tensors='pt',
    )

    # Concatenate prompts and responses
    full_sequences = [p + r for p, r in zip(prompt_texts, response_texts)]
    full_encodings = tokenizer(
        full_sequences,
        padding='max_length',
        max_length=max_prompt_length + max_response_length,
        truncation=True,
        return_tensors='pt',
    )

    return (
        prompt_encodings['input_ids'].to(device),
        prompt_encodings['attention_mask'].to(device),
        response_encodings['input_ids'].to(device),
        response_encodings['attention_mask'].to(device),
        full_encodings['input_ids'].to(device),
        full_encodings['attention_mask'].to(device),
    )
