import torch
import triton
from triton_flash_attn import flash_attention


def benchmark():
    if not torch.cuda.is_available():
        print("CUDA device required for benchmark.")
        return

    configs = [
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (2, 8, 4096, 64),
    ]

    for B, H, N, D in configs:
        q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
        v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

        ms = triton.testing.do_bench(lambda: flash_attention(q, k, v))
        tflops = (4 * B * H * (N ** 2) * D) / (ms * 1e-3) / 1e12

        print(f"Batch: {B}, Heads: {H}, SeqLen: {N}, HeadDim: {D} | Execution Time: {ms:.3f} ms | TFLOPS: {tflops:.2f}")


if __name__ == "__main__":
    benchmark()