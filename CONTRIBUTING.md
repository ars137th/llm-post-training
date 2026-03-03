# Contributing to LLM Post-Training

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Bug reports**: Open an issue describing the bug and how to reproduce it
- **Feature requests**: Suggest new techniques, models, or features
- **Code contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve docs, add examples, write tutorials
- **Testing**: Add test cases, report issues
- **Notebooks**: Create educational Jupyter notebooks

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/llm-post-training.git
   cd llm-post-training
   ```
3. **Set up development environment**:
   ```bash
   pip install -r requirements/dev.txt
   pip install -e .
   pre-commit install
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Before Making Changes

1. Check existing issues and PRs to avoid duplicating work
2. For major changes, open an issue first to discuss the approach
3. Ensure tests pass: `pytest tests/`

### Making Changes

1. **Write clean code**:
   - Follow PEP 8 style guide
   - Use type hints
   - Add docstrings for functions/classes
   - Keep functions focused and modular

2. **Format your code**:
   ```bash
   black src/ tests/ scripts/
   isort src/ tests/ scripts/
   ```

3. **Add tests** for new features:
   - Unit tests in `tests/`
   - Test edge cases and error handling
   - Aim for >80% code coverage

4. **Update documentation**:
   - Update docstrings
   - Update relevant docs in `docs/`
   - Add examples if appropriate
   - Update README if needed

### Committing Changes

1. **Write clear commit messages**:
   ```
   Add DPO loss function implementation

   - Implement Bradley-Terry loss
   - Add tests for preference ranking
   - Include docstrings with mathematical formulation
   ```

2. **Keep commits focused**: One logical change per commit

3. **Run checks before committing**:
   ```bash
   # Format check
   black --check src/ tests/ scripts/
   isort --check src/ tests/ scripts/

   # Type check
   mypy src/

   # Run tests
   pytest tests/
   ```

### Submitting a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issue(s)
   - Screenshots/examples if applicable
   - Checklist of completed items

3. **PR Checklist**:
   - [ ] Code follows style guidelines (black, isort)
   - [ ] Type hints added
   - [ ] Docstrings added/updated
   - [ ] Tests added/updated
   - [ ] Tests pass locally
   - [ ] Documentation updated
   - [ ] No merge conflicts

## Code Style

### Python Style

- **PEP 8** compliance (enforced by `black`)
- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Imports**: Organized with `isort`
- **Type hints**: Use for function signatures
- **Docstrings**: Google style

Example:
```python
def compute_dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute Direct Preference Optimization loss.

    Args:
        policy_logps: Log probabilities from policy model [batch_size, 2]
                     where [:,0] is chosen, [:,1] is rejected
        reference_logps: Log probabilities from reference model [batch_size, 2]
        beta: KL divergence coefficient (higher = stay closer to reference)

    Returns:
        Scalar loss value

    References:
        Rafailov et al., "Direct Preference Optimization", 2023
        https://arxiv.org/abs/2305.18290
    """
    # Implementation here
    ...
```

### Documentation

- **Docstrings**: Required for all public functions/classes
- **Comments**: Explain *why*, not *what* (code should be self-explanatory)
- **Type hints**: Help users understand function signatures
- **Examples**: Include usage examples in docstrings

### Testing

Tests should be:
- **Fast**: Use small models/datasets for testing
- **Isolated**: Each test independent of others
- **Clear**: Test names describe what's being tested

Example:
```python
def test_language_model_generation():
    """Test that LanguageModel can generate text."""
    model = LanguageModel.from_pretrained("gpt2", use_lora=False)
    processor = TextProcessor(model.tokenizer)

    input_text = "Hello"
    encoded = processor.tokenize(input_text)
    output = model.generate(encoded["input_ids"], max_new_tokens=5)

    assert output.shape[0] == 1
    assert output.shape[1] > encoded["input_ids"].shape[1]
```

## Areas for Contribution

### High Priority

1. **Core Implementations**:
   - [ ] SFT trainer implementation
   - [ ] Reward model training
   - [ ] PPO/RLHF implementation
   - [ ] DPO/IPO trainers
   - [ ] Evaluation metrics

2. **Multimodal Support**:
   - [ ] Vision-language model wrapper
   - [ ] Image preprocessing
   - [ ] Multimodal evaluation metrics

3. **Documentation**:
   - [ ] Jupyter notebooks for each technique
   - [ ] User guides
   - [ ] API documentation

### Medium Priority

4. **Additional Techniques**:
   - Constitutional AI
   - RLAIF (RL from AI Feedback)
   - Rejection sampling
   - Best-of-N sampling

5. **Optimizations**:
   - DeepSpeed integration
   - Flash Attention support
   - Gradient checkpointing
   - Memory profiling

6. **Evaluation**:
   - More benchmarks (MT-Bench, AlpacaEval)
   - Automated evaluation pipelines
   - Reward model evaluation tools

### Nice to Have

7. **Infrastructure**:
   - Docker containers
   - Example cloud deployment configs
   - CI/CD pipelines

8. **Tools**:
   - Dataset preparation scripts
   - Model merging utilities
   - Checkpoint conversion tools

## Review Process

1. **Automated checks**: GitHub Actions will run tests and linting
2. **Code review**: Maintainers will review code and provide feedback
3. **Iteration**: Address feedback, update PR
4. **Merge**: Once approved, PR will be merged

## Questions?

- **General questions**: Open a discussion on GitHub
- **Bug reports**: Open an issue
- **Feature requests**: Open an issue with "feature" label
- **Security issues**: Email directly (don't open public issue)

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to learn and build something useful together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

Thank you for contributing! 🎉
