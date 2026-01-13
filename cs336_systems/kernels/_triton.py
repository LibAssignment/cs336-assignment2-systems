from torch import Tensor
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
  X_ptr, Y_ptr,
  L_ptr,
  s_x0, s_x1,
  s_y0, s_y1,
  s_l0,
  B, N,
  N_TILE: tl.constexpr,
):
  batch_idx = tl.program_id(0)

  x_ptr = tl.make_block_ptr(
    X_ptr + batch_idx * s_x0,
    strides=(s_x1,),
    shape=(N,),
    offsets=(0,),
    block_shape=(N_TILE,),
    order=(0,),
  )

  y_ptr = tl.make_block_ptr(
    Y_ptr + batch_idx * s_y0,
    strides=(s_y1,),
    shape=(N,),
    offsets=(0,),
    block_shape=(N_TILE,),
    order=(0,),
  )

  l_ptr = tl.make_block_ptr(
    L_ptr + batch_idx * s_l0,
    strides=(1,),
    shape=(1,),
    offsets=(0,),
    block_shape=(1,),
    order=(0,),
  )

  lse = tl.full((1,), float("-inf"), dtype=tl.float32)
  for i in range(tl.cdiv(N, N_TILE)):
    x = tl.load(x_ptr, boundary_check=(0,), padding_option="nan") # TODO: should fill -inf
    x_max = tl.max(x, axis=-1, keep_dims=True)
    exps = tl.exp(x - x_max)
    sum_exps = tl.sum(exps, axis=-1, keep_dims=True)
    lse = tl.log(tl.exp(lse - x_max) + sum_exps) + x_max
    x_ptr = x_ptr.advance((N_TILE,))
  tl.store(l_ptr, lse, boundary_check=(0,))

  x_ptr = tl.make_block_ptr(
    X_ptr + batch_idx * s_x0,
    strides=(s_x1,),
    shape=(N,),
    offsets=(0,),
    block_shape=(N_TILE,),
    order=(0,),
  )

  for i in range(tl.cdiv(N, N_TILE)):
    x = tl.load(x_ptr, boundary_check=(0,), padding_option="zero")
    y = (x - lse).exp()
    tl.store(y_ptr, y, boundary_check=(0,))
    x_ptr = x_ptr.advance((N_TILE,))
    y_ptr = y_ptr.advance((N_TILE,))

def softmax_triton(X: Tensor, *, TILE: int = 16) -> tuple[Tensor, Tensor]:
  Y = torch.empty_like(X)
  old_shape = X.shape
  L = torch.empty((*old_shape[:-1], 1), device=X.device, dtype=X.dtype)
  X1 = X.reshape(-1, old_shape[-1])
  Y1 = Y.reshape(-1, old_shape[-1])
  L1 = L.reshape(-1)
  B, N = X1.shape

  grid = (B,)

  softmax_kernel[grid](
    X1,
    Y1,
    L1,
    X1.stride(0), X1.stride(1),
    Y1.stride(0), Y1.stride(1),
    L1.stride(0),
    B, N,
    TILE, # type: ignore
  )

  return Y, L
