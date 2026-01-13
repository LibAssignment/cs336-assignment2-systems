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
    output = exps / sum_exps
    ctx.save_for_backward(output)
    setattr(ctx, "dim", dim)
    return output
