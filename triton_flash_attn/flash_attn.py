import torch
import triton

from .kernels.flash_attn_kernel import _flash_attn_fwd_kernel


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float):
        BLOCK_M = 64
        BLOCK_N = 64

        batch_size, num_heads, seq_len, head_dim = q.shape

        out = torch.empty_like(q)

        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads, 1)

        _flash_attn_fwd_kernel[grid](
            q, k, v, sm_scale,
            None, None,
            out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch_size, num_heads, seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=head_dim,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float = None) -> torch.Tensor:
    """
    FlashAttention autograd launcher.
    Expects tensors of shape (Batch, Heads, Seq_Len, Head_Dim) in float16.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return FlashAttentionFunction.apply(q, k, v, sm_scale)