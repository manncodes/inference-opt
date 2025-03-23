# Inference-Opt

Implementation of advanced inference optimization techniques for modern language models:
- **Radix Attention**: Efficient KV cache reuse during runtime
- **Multi-head Latent Attention (MLA)**: DeepSeek's KV cache compression technique
- **Speculative Decoding**: Accelerated inference through parallel token prediction

## Overview

This repository provides modular implementations of three cutting-edge techniques for optimizing transformer inference. Each component can be used independently or combined for maximum efficiency:

1. **Radix Attention**: Automatically detects and reuses computations from the KV cache, compatible with continuous batching and paged attention.
2. **Multi-head Latent Attention (MLA)**: Implements DeepSeek's approach for compressing the KV cache into a low-dimensional latent space, reducing memory requirements while maintaining or improving performance.
3. **Speculative Decoding**: Uses a smaller, efficient "drafter" model to predict multiple tokens in parallel, which are then verified by the larger target model, enabling significant speedups without changing outputs.

## Installation

```bash
git clone https://github.com/manncodes/inference-opt.git
cd inference-opt
pip install -e .
```

## Usage

### Basic Example

```python
import torch
from inference_opt import TransformerWithOptimizations

# Initialize model with all optimizations
model = TransformerWithOptimizations(
    dim=512,
    depth=6,
    heads=8,
    use_radix_attention=True,
    use_mla=True,
    use_speculative_decoding=True,
    mla_latent_dim=64,
    speculative_model=None  # Will create a smaller version automatically
)

# Generate tokens from a prompt
input_ids = torch.tensor([[1, 2, 3, 4]])  # Example input sequence
output = model.generate(input_ids, max_length=100)
```

### Ablation Studies

The package is designed to enable easy ablation studies to measure the impact of each optimization:

```python
from inference_opt.experiment import run_ablation_study

results = run_ablation_study(
    input_dataset="path/to/dataset",
    model_config={
        "dim": 512,
        "depth": 6,
        "heads": 8,
    },
    techniques_to_test=[
        "radix_attention",
        "mla",
        "speculative_decoding",
    ],
    metrics=["throughput", "memory_usage", "quality"]
)
```

## Components

### Radix Attention

Our implementation of Radix Attention follows the approach described in SGLang, enabling efficient KV cache reuse during runtime. This significantly improves inference efficiency, especially for repetitive patterns in the input.

### Multi-head Latent Attention (MLA)

MLA, proposed by DeepSeek, compresses the attention input into a low-dimensional latent vector, which reduces the memory required for KV cache while maintaining model quality. Our implementation includes:
- Compression of input vectors into a latent space
- Decoupled Rotary Positional Embeddings
- Efficient latent vector storage and projection

### Speculative Decoding

Speculative decoding accelerates inference by:
- Using a smaller "drafter" model to predict multiple tokens in parallel
- Verifying predictions with the target model
- Accepting correct predictions to skip multiple decoding steps

## References

- RadixAttention: [SGLang Blog](https://lmsys.org/blog/2024-01-17-sglang/), [Paper](https://arxiv.org/abs/2312.07104)
- Multi-head Latent Attention: [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- Speculative Decoding: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), [Accelerating Large Language Model Decoding](https://arxiv.org/abs/2302.01318)

## License

MIT
