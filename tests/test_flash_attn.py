import pytest
import torch
from triton_flash_attn import flash_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attn_correctness():
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 256, 64

    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

    # Triton FlashAttention
    out_triton = flash_attention(q, k, v)

    # Reference PyTorch PyTorch Multi-head / Causal Attention
    sm_scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    causal_mask = torch.triu(torch.ones((N, N), device="cuda"), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(attn_weights, v)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)