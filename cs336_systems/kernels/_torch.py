# %%
from torch import Tensor
import torch
from torch.autograd.function import Function, FunctionCtx

class SoftmaxTorch(Function):
  @staticmethod
  def forward(ctx: FunctionCtx, input: Tensor, dim: int = -1):
    max_input = input.amax(dim=dim, keepdim=True)
    exps = (input - max_input).exp()
    sum_exps = exps.sum(dim=dim, keepdim=True)
    lse = sum_exps.log() + max_input
    output = exps / sum_exps
    ctx.save_for_backward(lse, output)
    setattr(ctx, "dim", dim)
    return output

class SoftmaxTorchTile(Function):
  @staticmethod
  def forward(ctx: FunctionCtx, X: Tensor):
    BLOCK = 16
    if X.shape[-1] < BLOCK:
      BLOCK = X.shape[-1]
    assert X.shape[-1] % BLOCK == 0, "Input size must be divisible by TILE"

    lse = torch.full((*X.shape[:-1], 1,), float("-inf"), dtype=X.dtype, device=X.device)
    Y = torch.empty_like(X)
    for i in range(X.shape[-1] // BLOCK):
      X_i = X[..., i*BLOCK:(i+1)*BLOCK]
      X_max = X_i.amax(dim=-1, keepdim=True)
      exps = (X_i - X_max).exp()
      sum_exps = exps.sum(dim=-1, keepdim=True)
      lse[...] = ((lse - X_max).exp() + sum_exps).log() + X_max

    for i in range(X.shape[-1] // BLOCK):
      X_i = X[..., i*BLOCK:(i+1)*BLOCK]
      Y_i = (X_i - lse).exp()
      Y[..., i*BLOCK:(i+1)*BLOCK] = Y_i

    # print(lse)
    ctx.save_for_backward(lse, Y)
    setattr(ctx, "dim", -1)
    return Y

class SoftmaxTriton(Function):
  @staticmethod
  def forward(ctx: FunctionCtx, input: Tensor):
    from ._triton import softmax_triton

    output, lse = softmax_triton(input)
    # print(lse)
    ctx.save_for_backward(lse, output)
    setattr(ctx, "dim", -1)
    return output
