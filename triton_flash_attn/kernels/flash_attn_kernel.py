import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused Causal FlashAttention Forward Pass Kernel in Triton.
    Uses online softmax updating over blocks of size BLOCK_M x BLOCK_N.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute base memory pointer offsets for Q, K, V
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Online softmax accumulators initialized in float32 for high precision
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q block once per program instance
    q = tl.load(Q + off_q, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Loop over KV tile blocks up to the current query block (causal mask)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(K + off_k + start_n * stride_kn, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)

        # Matmul: Q @ K.T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Causal mask application
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))

        # Online Softmax updates
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Matmul: Softmax(P) @ V
        v = tl.load(V + off_v + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_new

    # Normalization by sum of weights
    acc = acc / l_i[:, None]

    # Store final computed output
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(Out + off_o, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)