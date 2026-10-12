# triton-flash-attn
Triton-based custom kernel library specializing in FlashAttention and context-scaling operations
# Triton Custom Kernels

Contributed FlashAttention extensions and custom CUDA kernels for hardware-efficient context scaling.

## Installation

```bash
pip install -e .
```
## Running Tests
```bash
pytest tests/
```
## Running Benchmarks
```bash
python benchmarks/bench_flash_attn.py
```