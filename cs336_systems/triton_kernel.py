import triton
import triton.language as tl
import torch

@triton.jit
def attention_kernel(
  Q_ptr, K_ptr, V_ptr,
  O_ptr, L_ptr,
  s_q0, s_qs, s_qd,
  s_k0, s_ks, s_kd,
  s_v0, s_vs, s_vd,
  s_o0, s_os, s_od,
  s_l0, s_ls,
  Nq, Nk,
  scale, # scale is 1/sqrt(d_k)
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
  is_causal: tl.constexpr,
):
  q_seq_idx = tl.program_id(0)
  batch_idx = tl.program_id(1)

  q_stride = (s_q0, s_qs, s_qd)
  k_stride = (s_k0, s_ks, s_kd)
  v_stride = (s_v0, s_vs, s_vd)
  o_stride = (s_o0, s_os, s_od)
  l_stride = (s_l0, s_ls)
  q_ptr = tl.make_block_ptr(
    Q_ptr + batch_idx * q_stride[0],
    strides=q_stride[-2:],
    shape=(Nq, D),
    offsets=(q_seq_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
  )
  o_ptr = tl.make_block_ptr(
    O_ptr + batch_idx * o_stride[0],
    strides=o_stride[-2:],
    shape=(Nq, D),
    offsets=(q_seq_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
  )
  l_ptr = tl.make_block_ptr(
    L_ptr + batch_idx * l_stride[0],
    strides=l_stride[-1:],
    shape=(Nq,),
    offsets=(q_seq_idx * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
  )
  Q_i = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
  O_i = tl.zeros((Q_TILE_SIZE, D), tl.float32) # saves (x - x_max).exp() @ V
  D_i = tl.full((Q_TILE_SIZE,), float('-inf'), tl.float32) # saves x_max
  L_i = tl.zeros((Q_TILE_SIZE,), tl.float32) # saves (x - x_max).exp().sum()

  k_seq_max = tl.cdiv(Nk, K_TILE_SIZE)
  if is_causal:
    k_seq_max = tl.cdiv((q_seq_idx + 1) * Q_TILE_SIZE, K_TILE_SIZE)

  for k_seq_idx in range(k_seq_max):
    k_ptr = tl.make_block_ptr(
      K_ptr + batch_idx * k_stride[0],
      strides=k_stride[-2:],
      shape=(Nk, D),
      offsets=(k_seq_idx * K_TILE_SIZE, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
    )

    v_ptr = tl.make_block_ptr(
      V_ptr + batch_idx * v_stride[0],
      strides=v_stride[-2:],
      shape=(Nk, D),
      offsets=(k_seq_idx * K_TILE_SIZE, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
    )

    K_ij = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
    V_ij = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")

    # Compute QK^T
    # Q_TILE_SIZE x K_TILE_SIZE
    S = tl.dot(Q_i, tl.trans(K_ij), out_dtype=tl.float32) * scale

    if is_causal:
      q_indices = tl.arange(0, Q_TILE_SIZE) + q_seq_idx * Q_TILE_SIZE
      k_indices = tl.arange(0, K_TILE_SIZE) + k_seq_idx * K_TILE_SIZE
      mask = q_indices[:, None] >= k_indices[None, :]
      S = tl.where(mask, S, float('-inf'))

    # softmax of s
    x_max = tl.maximum(D_i, tl.max(S, axis=-1)) # Q_TILE_SIZE
    x_exp = tl.exp(S - x_max[:, None]) # Q_TILE_SIZE x K_TILE_SIZE
    x_exp_sum = tl.sum(x_exp, axis=-1) # Q_TILE_SIZE

    # O = P V
    o = tl.dot(x_exp, V_ij) # Q_TILE_SIZE x D

    # accumulate
    exp_delta = tl.exp(D_i - x_max) # Q_TILE_SIZE
    L_i = L_i * exp_delta + x_exp_sum # Q_TILE_SIZE
    O_i = O_i * exp_delta[:, None] + o # Q_TILE_SIZE x D
    D_i = x_max # Q_TILE_SIZE

  O_i = O_i / L_i[:, None]
  L_i = tl.log(L_i) + D_i
  tl.store(o_ptr, O_i, boundary_check=(0, 1))
  tl.store(l_ptr, L_i, boundary_check=(0,))

def _flash_attention_triton_forward(
  Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
  *,
  is_causal=False,
  Q_TILE_SIZE = 16,
  K_TILE_SIZE = 64,
):
  assert Q.dim() == 3
  assert K.dim() == 3
  assert V.dim() == 3
  assert Q.shape[0] == K.shape[0] == V.shape[0], "Batch size must be the same"
  assert Q.shape[2] == K.shape[2] == V.shape[2], "Embedding dimension must be the same"
  assert K.shape[1] == V.shape[1], "Sequence length of K and V must be the same"

  B, Nq, d_k = Q.shape
  Nk = K.shape[1]

  O = torch.empty((B, Nq, d_k), device=Q.device, dtype=Q.dtype)
  L = torch.empty((B, Nq), device=Q.device, dtype=Q.dtype)

  scale = 1.0 / (d_k ** 0.5)

  grid = (triton.cdiv(Nq, Q_TILE_SIZE), B)

  attention_kernel[grid](
    Q, K, V,
    O, L,
    Q.stride(-3), Q.stride(-2), Q.stride(-1),
    K.stride(-3), K.stride(-2), K.stride(-1),
    V.stride(-3), V.stride(-2), V.stride(-1),
    O.stride(-3), O.stride(-2), O.stride(-1),
    L.stride(-2), L.stride(-1),
    Nq, Nk,
    scale,
    D=d_k, # type: ignore
    Q_TILE_SIZE=Q_TILE_SIZE, # type: ignore
    K_TILE_SIZE=K_TILE_SIZE, # type: ignore
    is_causal=is_causal, # type: ignore
  )

  return O, L

# %%
@triton.jit
def attention_kernel_backward(
  L_ptr, Q_ptr, K_ptr, V_ptr, O_ptr,
  dO_ptr,
  dQ_ptr, dK_ptr, dV_ptr,
  s_l0, s_ls,
  s_q0, s_qs, s_qd,
  s_k0, s_ks, s_kd,
  s_v0, s_vs, s_vd,
  s_o0, s_os, s_od,
  s_do0, s_dos, s_dod,
  s_dq0, s_dqs, s_dqd,
  s_dk0, s_dks, s_dkd,
  s_dv0, s_dvs, s_dvd,
  Nq, Nk,
  scale,
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
  is_causal: tl.constexpr,
):
  q_seq_idx = tl.program_id(0)
  batch_idx = tl.program_id(1)

  l_stride = (s_l0, s_ls)
  q_stride = (s_q0, s_qs, s_qd)
  k_stride = (s_k0, s_ks, s_kd)
  v_stride = (s_v0, s_vs, s_vd)
  o_stride = (s_o0, s_os, s_od)
  do_stride = (s_do0, s_dos, s_dod)

  dq_stride = (s_dq0, s_dqs, s_dqd)
  dk_stride = (s_dk0, s_dks, s_dkd)
  dv_stride = (s_dv0, s_dvs, s_dvd)

  l_ptr = tl.make_block_ptr(
    L_ptr + batch_idx * l_stride[0],
    strides=l_stride[-1:],
    shape=(Nq,),
    offsets=(q_seq_idx * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
  )
  q_ptr = tl.make_block_ptr(
    Q_ptr + batch_idx * q_stride[0],
    strides=q_stride[-2:],
    shape=(Nq, D),
    offsets=(q_seq_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
  )
  o_ptr = tl.make_block_ptr(
    O_ptr + batch_idx * o_stride[0],
    strides=o_stride[-2:],
    shape=(Nq, D),
    offsets=(q_seq_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
  )
  do_ptr = tl.make_block_ptr(
    dO_ptr + batch_idx * do_stride[0],
    strides=do_stride[-2:],
    shape=(Nq, D),
    offsets=(q_seq_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
  )

  dq_ptr = tl.make_block_ptr(
    dQ_ptr + batch_idx * dq_stride[0],
    strides=dq_stride[-2:],
    shape=(Nq, D),
    offsets=(q_seq_idx * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
  )

  L_i = tl.load(l_ptr, boundary_check=(0,), padding_option="zero")
  Q_i = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")
  O_i = tl.load(o_ptr, boundary_check=(0, 1), padding_option="zero")
  dO_i = tl.load(do_ptr, boundary_check=(0, 1), padding_option="zero")

  dQ_i = tl.zeros((Q_TILE_SIZE, D), tl.float32)
  D_i = tl.sum(O_i * dO_i, axis=-1)[:, None]  # Q_TILE_SIZE, 1
  L_i = L_i[:, None]  # Q_TILE_SIZE, 1

  k_seq_max = tl.cdiv(Nk, K_TILE_SIZE)
  if is_causal:
    k_seq_max = tl.cdiv((q_seq_idx + 1) * Q_TILE_SIZE, K_TILE_SIZE)

  for k_seq_idx in range(k_seq_max):
    k_ptr = tl.make_block_ptr(
      K_ptr + batch_idx * k_stride[0],
      strides=k_stride[-2:],
      shape=(Nk, D),
      offsets=(k_seq_idx * K_TILE_SIZE, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
    )

    v_ptr = tl.make_block_ptr(
      V_ptr + batch_idx * v_stride[0],
      strides=v_stride[-2:],
      shape=(Nk, D),
      offsets=(k_seq_idx * K_TILE_SIZE, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
    )

    dk_ptr = tl.make_block_ptr(
      dK_ptr + batch_idx * dk_stride[0],
      strides=dk_stride[-2:],
      shape=(Nk, D),
      offsets=(k_seq_idx * K_TILE_SIZE, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
    )

    dv_ptr = tl.make_block_ptr(
      dV_ptr + batch_idx * dv_stride[0],
      strides=dv_stride[-2:],
      shape=(Nk, D),
      offsets=(k_seq_idx * K_TILE_SIZE, 0),
      block_shape=(K_TILE_SIZE, D),
      order=(1, 0),
    )

    K_ij = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
    V_ij = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")

    # Compute QK^T
    # Q_TILE_SIZE x K_TILE_SIZE
    S = tl.dot(Q_i, tl.trans(K_ij), out_dtype=tl.float32) * scale

    if is_causal:
      q_indices = tl.arange(0, Q_TILE_SIZE) + q_seq_idx * Q_TILE_SIZE
      k_indices = tl.arange(0, K_TILE_SIZE) + k_seq_idx * K_TILE_SIZE
      mask = q_indices[:, None] >= k_indices[None, :]
      S = tl.where(mask, S, float('-inf'))

    # softmax of s
    P = tl.exp(S - L_i) * scale # Q_TILE_SIZE x K_TILE_SIZE

    dV_ij = tl.dot(tl.trans(P), dO_i) # K_TILE_SIZE x D
    dP = tl.dot(dO_i, tl.trans(V_ij)) # Q_TILE_SIZE x K_TILE_SIZE
    dS_ij = P * (dP - D_i) * scale # Q_TILE_SIZE x K_TILE_SIZE
    dQ_ij = tl.dot(dS_ij, K_ij) # Q_TILE_SIZE x D
    dK_ij = tl.dot(tl.trans(dS_ij), Q_i) # K_TILE_SIZE x D
    # accumulate

    dQ_i += dQ_ij
    tl.atomic_add(dk_ptr, dK_ij, ) # TODO: boundary_check here?
    tl.atomic_add(dv_ptr, dV_ij, )
  tl.store(dq_ptr, dQ_i, boundary_check=(0, 1))


def _flash_attention_triton_backward(
  L: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
  dO: torch.Tensor,
  *,
  is_causal=False,
  Q_TILE_SIZE = 16,
  K_TILE_SIZE = 64,
):
  assert L.dim() == 2
  assert Q.dim() == 3
  assert K.dim() == 3
  assert V.dim() == 3
  assert O.dim() == 3
  assert L.shape[0] == Q.shape[0] == K.shape[0] == V.shape[0] == O.shape[0], "Batch size must be the same"
  assert Q.shape[2] == K.shape[2] == V.shape[2], "Embedding dimension must be the same"
  assert L.shape[1] == Q.shape[1] == O.shape[1], "Sequence length of L, Q, and O must be the same"
  assert K.shape[1] == V.shape[1], "Sequence length of K and V must be the same"
  B, Nq, d_k = Q.shape
  Nk = K.shape[1]

  dQ = torch.zeros_like(Q)
  dK = torch.zeros_like(K)
  dV = torch.zeros_like(V)

  scale = 1.0 / (d_k ** 0.5)

  grid = (triton.cdiv(Nq, Q_TILE_SIZE), B)

  attention_kernel_backward[grid](
    L, Q, K, V, O,
    dO,
    dQ, dK, dV,
    L.stride(-2), L.stride(-1),
    Q.stride(-3), Q.stride(-2), Q.stride(-1),
    K.stride(-3), K.stride(-2), K.stride(-1),
    V.stride(-3), V.stride(-2), V.stride(-1),
    O.stride(-3), O.stride(-2), O.stride(-1),
    dO.stride(-3), dO.stride(-2), dO.stride(-1),
    dQ.stride(-3), dQ.stride(-2), dQ.stride(-1),
    dK.stride(-3), dK.stride(-2), dK.stride(-1),
    dV.stride(-3), dV.stride(-2), dV.stride(-1),
    Nq, Nk,
    scale,
    D=d_k, # type: ignore
    Q_TILE_SIZE=Q_TILE_SIZE, # type: ignore
    K_TILE_SIZE=K_TILE_SIZE, # type: ignore
    is_causal=is_causal, # type: ignore
  )

  return dQ, dK, dV

# %%
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
  x_ptr, weight_ptr, # Input pointers
  output_ptr, # Output pointer
  x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
  weight_stride_dim, # Likely 1
  output_stride_row, # Likely 1
  ROWS, D,
  ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Tile shapes must be known at compile time
):
  row_tile_idx = tl.program_id(0)

  x_block_ptr = tl.make_block_ptr(
    x_ptr,
    shape=(ROWS, D),
    strides=(x_stride_row, x_stride_dim),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
  )

  weight_block_ptr = tl.make_block_ptr(
    weight_ptr,
    shape=(D),
    strides=(weight_stride_dim,),
    offsets=(0,),
    block_shape=(D_TILE_SIZE,),
    order=(0,),
  )

  output_block_ptr = tl.make_block_ptr(
    output_ptr,
    shape=(ROWS,),
    strides=(output_stride_row,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
  )

  output = tl.zeros((ROWS_TILE_SIZE,), tl.float32)

  for i in range(tl.cdiv(D, D_TILE_SIZE)):
    row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
    output += tl.sum(row * weight[None, :], axis=1)
    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
    weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

  tl.store(output_block_ptr, output, boundary_check=(0,))

import torch
def _weighted_sum(
  x: torch.Tensor,
  weight: torch.Tensor,
):
  assert x.dim() == 2
  assert weight.dim() == 1
  assert x.shape[1] == weight.shape[0]

  ROWS, D = x.shape
  output = torch.empty((ROWS,), device=x.device, dtype=torch.float32)

  grid = lambda meta: (triton.cdiv(ROWS, meta['ROWS_TILE_SIZE']), )

  weighted_sum_fwd[grid](
    x,
    weight,
    output,
    x.stride(0), x.stride(1),
    weight.stride(0),
    output.stride(0),
    ROWS, D,
    ROWS_TILE_SIZE=32, # type: ignore
    D_TILE_SIZE=64, # type: ignore
  )

  return output

# %%
# _weighted_sum(torch.randn(128, 256, device="cuda"), torch.randn(256, device="cuda"))
# %%
