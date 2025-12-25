# %%
import torch.cuda.nvtx as nvtx
from einops import einsum
import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd.function import Function, FunctionCtx

class FlashAttnVanilla(Function):
  """
  Vanilla PyTorch implementation that works like FlashAttention2.

  - doesn't take mask for simplicity
  - saves log-sum-exp tensor for backward pass
  - might save memory by recomputing
  """

  @staticmethod
  def forward(ctx: FunctionCtx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
    # TODO: warning?
    O, L = _vanilla_flash_attention_forward(Q, K, V, is_causal)
    L = L.squeeze(-1)  # remove last dim
    ctx.save_for_backward(L, Q, K, V, O)
    return O

  @staticmethod
  def backward(ctx, *grad_outputs):
    raise NotImplementedError

class FlashAttnTorch(Function):
  """
  FlashAttention2 implemented using only standard PyTorch operations.

  Q_i, K_j, V_j => O_ij, L_ij, D_ij

  L = logsumexp(x)
  D = max(x)
  """

  @staticmethod
  def forward(ctx: FunctionCtx, Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
    BLOCK_Q = 16
    BLOCK_K = 64
    assert Q.shape[-2] % BLOCK_Q == 0, "FlashAttnTorch only supports sequence length divisible by BLOCK_SIZE"
    assert K.shape[-2] % BLOCK_K == 0, "FlashAttnTorch only supports sequence length divisible by BLOCK_SIZE"
    assert K.shape[-2] == V.shape[-2], "FlashAttnTorch requires K and V to have the same sequence length"

    Os = []
    Ls = []
    for i in range(Q.shape[-2] // BLOCK_Q):
      Q_i = Q[..., i*BLOCK_Q:(i+1)*BLOCK_Q, :]
      O_i = torch.zeros((*Q_i.shape[:-1], V.shape[-1]), device=Q.device, dtype=Q.dtype)
      L_i = torch.zeros((*Q_i.shape[:-1], 1), device=Q.device, dtype=Q.dtype)

      for j in range(K.shape[-2] // BLOCK_K):
        K_j = K[..., j*BLOCK_K:(j+1)*BLOCK_K, :]
        V_j = V[..., j*BLOCK_K:(j+1)*BLOCK_K, :]

        O_ij, L_ij = _vanilla_flash_attention_forward(Q_i, K_j, V_j, is_causal)
        L_new = (1 + (L_ij - L_i).exp()).log() + L_i
        O_i = O_i * (L_i - L_new).exp() + O_ij * (L_ij - L_new).exp()
        L_i = L_new
      assert O_i is not None and L_i is not None, "No blocks were processed in FlashAttnTorch forward pass."
      Os.append(O_i)
      Ls.append(L_i)
    O = torch.cat(Os, dim=-2)
    L = torch.cat(Ls, dim=-2).squeeze(-1)  # remove last dim
    assert O.shape == (*Q.shape[:-1], V.shape[-1]), "Output shape mismatch in FlashAttnTorch forward pass. from {} {}, got {}".format(Q.shape, V.shape, O.shape)
    assert L.shape == Q.shape[:-1], "Log-sum-exp shape mismatch in FlashAttnTorch forward pass. from {} got {}".format(Q.shape, L.shape)
    ctx.save_for_backward(L, Q, K, V, O)
    return O

  @staticmethod
  def backward(ctx, *grad_outputs):
    return FlashAttnVanilla.backward(ctx, *grad_outputs)


def _vanilla_flash_attention_forward(Q: Tensor, K: Tensor, V: Tensor, is_causal=False):
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
    L = L.log() + D
  with nvtx.range("attention_weighted_sum"):
    result = einsum(atten, V, "... queries keys, ... keys d_v -> ... queries d_v")
  return result, L
