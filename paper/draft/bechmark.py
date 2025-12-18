# %%
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

def step(x: Tensor, y: Tensor, backward: bool = True) -> float:
  y_hat = llm(x) # type: Tensor
  loss = _cross_entory(y_hat, y).mean()
  if backward:
    optim.zero_grad()
    # starttime = timeit.default_timer()
    loss.backward()
    torch.cuda.synchronize()
    # endtime = timeit.default_timer()
    # print(f"Backward time: {endtime - starttime:.4f} seconds")
    optim.step()
  torch.cuda.synchronize()
  return loss.item()

# %%
for epoch in range(10):
  step(x, y)

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
