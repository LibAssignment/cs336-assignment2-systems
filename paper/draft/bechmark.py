# %%
import torch.cuda.nvtx as nvtx
import timeit
from cs336_basics.config import Config

config = Config(vocab_size=10000)
llm, optim = config.create_llm(device="cuda")

# %%
from cs336_basics.modules import _cross_entory
import torch
from torch import Tensor
x = (torch.randn((config.batch_size, config.context_length + 1), device="cuda") % 1 * config.vocab_size).abs().long()
x, y = x[:, :-1], x[:, 1:]

def step(x: Tensor, y: Tensor, backward: bool = True) -> Tensor:
  with nvtx.range("forward"):
    y_hat = llm(x) # type: Tensor
    loss = _cross_entory(y_hat, y).mean()
  if backward:
    with nvtx.range("optim.zero_grad"):
      optim.zero_grad()
    # starttime = timeit.default_timer()
    with nvtx.range("backward"):
      loss.backward()
    # torch.cuda.synchronize()
    # endtime = timeit.default_timer()
    # print(f"Backward time: {endtime - starttime:.4f} seconds")
    with nvtx.range("optim.step"):
      optim.step()
  torch.cuda.synchronize()
  return loss


# %%
"""
# %%
for epoch in range(30):
  step(x, y)
%timeit step(x, y, backward=False) # type: ignore
%timeit step(x, y, backward=True) # type: ignore
%timeit step(x, y, backward=True) # type: ignore
%timeit step(x, y, backward=False) # type: ignore

# %%
%timeit -n10 step(x, y, backward=False) # type: ignore
%timeit -n20 step(x, y, backward=False) # type: ignore

# %%
%timeit -n10 step(x, y, backward=True) # type: ignore
%timeit -n20 step(x, y, backward=True) # type: ignore

# %%
"""

# %%
from einops import einsum
import torch
from jaxtyping import Bool, Float
from torch import Tensor
import cs336_basics.modules

@nvtx.range("scaled_dot_product_attention")
def _scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
  # TODO: warning?
  d_k = torch.tensor(K.shape[-1])
  with nvtx.range("attention_scores"):
    atten = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
  if mask is not None:
    with nvtx.range("attention_masking"):
      atten = atten.masked_fill(~mask, -torch.inf)
  with nvtx.range("attention_softmax"):
    atten = cs336_basics.modules._softmax(atten / d_k.sqrt(), dim=-1)
  with nvtx.range("attention_weighted_sum"):
    result = einsum(atten, V, "... queries keys, ... keys d_v -> ... queries d_v")
  return result
cs336_basics.modules._scaled_dot_product_attention = _scaled_dot_product_attention

# %%
for epoch in range(30):
  step(x, y)
