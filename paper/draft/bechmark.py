# %%
from typing import Any
import torch.cuda.nvtx as nvtx
from cs336_basics.config import Config

config = Config(vocab_size=10000)

def insert_nvtx(moduel: Any, name: str):
  import functools
  func = getattr(moduel, name)
  def wrapper(*args, **kwargs):
    with nvtx.range(name):
      return func(*args, **kwargs)
  functools.update_wrapper(wrapper, func)
  setattr(moduel, name, wrapper)
  return wrapper

import cs336_basics.modules
insert_nvtx(cs336_basics.modules, "_cross_entory")
insert_nvtx(cs336_basics.modules, "_softmax")

# %%
from cs336_basics.modules import _cross_entory
from dataclasses import dataclass, field
import timeit
import numpy as np
import torch
from torch import Tensor
x = (torch.randn((config.batch_size, config.context_length + 1), device="cuda") % 1 * config.vocab_size).abs().long()
x, y = x[:, :-1], x[:, 1:]

@dataclass
class Timing:
  starttime: float = field(default_factory=timeit.default_timer)
  forward_endtime: float = 0.0
  backward_endtime: float = 0.0
  optim_endtime: float = 0.0
  final_endtime: float = 0.0

  def mark_forward_end(self):
    self.forward_endtime = timeit.default_timer()
  def mark_backward_end(self):
    self.backward_endtime = timeit.default_timer()
  def mark_optim_end(self):
    self.optim_endtime = timeit.default_timer()
  def mark_final_end(self):
    self.final_endtime = timeit.default_timer()

  @staticmethod
  def report(timings: list["Timing"], unit: str = "seconds"):
    forward_times = np.array([t.forward_endtime - t.starttime for t in timings])
    backward_times = np.array([t.backward_endtime - t.forward_endtime for t in timings if t.backward_endtime > 0])
    optim_times = np.array([t.optim_endtime - t.backward_endtime for t in timings if t.optim_endtime > 0 and t.backward_endtime > 0])
    total_times = np.array([t.final_endtime - t.starttime for t in timings])

    def format_unit(seconds: float, unit: str) -> str:
      if unit == "seconds":
        return f"{seconds:.4f} s"
      elif unit == "milliseconds":
        return f"{seconds * 1000:.3f} ms"
      elif unit == "microseconds":
        return f"{seconds * 1_000_000:.3f} µs"
      else:
        return f"{seconds:.6f} s"

    print(f"Forward: {format_unit(forward_times.mean(), unit)} ± {format_unit(forward_times.std(), unit)}")
    if len(backward_times) > 0:
      print(f"Backward: {format_unit(backward_times.mean(), unit)} ± {format_unit(backward_times.std(), unit)}")
    if len(optim_times) > 0:
      print(f"Optim: {format_unit(optim_times.mean(), unit)} ± {format_unit(optim_times.std(), unit)}")
    print(f"Total: {format_unit(total_times.mean(), unit)} ± {format_unit(total_times.std(), unit)}")

def step(llm: torch.nn.Module, optim: torch.optim.Optimizer, x: Tensor, y: Tensor, backward: bool = True, optim_step: bool = True, timeit = False) -> tuple[Tensor, Timing | None]:
  timer = Timing() if timeit else None
  with nvtx.range("forward"):
    y_hat = llm(x) # type: Tensor
    loss = _cross_entory(y_hat, y).mean()
  if timer is not None:
    torch.cuda.synchronize()
    timer.mark_forward_end()
  if backward:
    if optim_step:
      with nvtx.range("optim.zero_grad"):
        optim.zero_grad()
    with nvtx.range("backward"):
      loss.backward()
    if timer is not None:
      torch.cuda.synchronize()
      timer.mark_backward_end()
    # torch.cuda.synchronize()
    # endtime = timer is not None.default_timer()
    # print(f"Backward time: {endtime - starttime:.4f} seconds")
    if optim_step:
      with nvtx.range("optim.step"):
        optim.step()
      if timer is not None:
        torch.cuda.synchronize()
        timer.mark_optim_end()
  torch.cuda.synchronize()
  if timer is not None:
    timer.mark_final_end()
  return loss, timer

# %%
def timeit_steps(llm: torch.nn.Module, optim: torch.optim.Optimizer, x: Tensor, y: Tensor, steps: int = 10, backward: bool = True, optim_step: bool = True):
  result = [step(llm, optim, x, y, backward=backward, optim_step=optim_step, timeit=True)[1] for _ in range(steps)]
  Timing.report([r for r in result if r is not None], unit="milliseconds")

if False:
  # %%
  print("forward pass:")
  llm, optim = config.create_llm(device="cuda")
  timeit_steps(llm, optim, x, y, steps=10, backward=False)
  timeit_steps(llm, optim, x, y, steps=20, backward=False)

  # %%
  print("backward pass:")
  llm, optim = config.create_llm(device="cuda")
  timeit_steps(llm, optim, x, y, steps=10, backward=True, optim_step=False)
  timeit_steps(llm, optim, x, y, steps=20, backward=True, optim_step=False)

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
if __file__ == "__main__":
  llm, optim = config.create_llm(device="cuda")
  for epoch in range(30):
    step(llm, optim, x, y, backward=True, optim_step=True, timeit=True)

# %%
