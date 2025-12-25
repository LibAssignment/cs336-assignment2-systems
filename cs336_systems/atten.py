# %%
import torch.cuda.nvtx as nvtx
from einops import einsum
import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd.function import Function, FunctionCtx

class FlashAttnTorch(Function):
  @staticmethod
  def forward(ctx: FunctionCtx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
    # TODO: warning?
    d_k = torch.tensor(K.shape[-1])
    with nvtx.range("attention_scores"):
      atten = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / d_k.sqrt()
    # if mask is not None:
    #   with nvtx.range("attention_masking"):
    #     atten = atten.masked_fill(~mask, -torch.inf)
    with nvtx.range("attention_softmax"):
      D = torch.max(atten, dim=-1, keepdim=True).values
      atten = (atten - D).exp()
      L = atten.sum(dim=-1, keepdim=True)
      atten = atten / L
      L = (L.log() + D).squeeze(-1)
    with nvtx.range("attention_weighted_sum"):
      result = einsum(atten, V, "... queries keys, ... keys d_v -> ... queries d_v")
    ctx.save_for_backward(L, Q, K, V, result)
    return result

  @staticmethod
  def backward(ctx, *grad_outputs):
    raise NotImplementedError
