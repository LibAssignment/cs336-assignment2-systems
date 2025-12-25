import triton
import triton.language as tl

@triton.jit
def attention_kernel(
  Q_ptr, K_ptr, V_ptr,
  O_ptr, L_ptr,
  s_q3, s_q2, s_q1,
  s_k3, s_k2, s_k1,
  s_v3, s_v2, s_v1,
  s_o3, s_o2, s_o1,
  s_l2, s_l1,
  Nq, Nk,
  scale,
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
):
  pass

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
    ROWS_TILE_SIZE=32,
    D_TILE_SIZE=64,
  )

  return output

# %%
# _weighted_sum(torch.randn(128, 256, device="cuda"), torch.randn(256, device="cuda"))
# %%
