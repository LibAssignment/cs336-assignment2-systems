# %%
from typing import Any
from omegaconf import OmegaConf
import torch.cuda.nvtx as nvtx

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
from cs336_basics.config import LLMConfig
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class TimingConfig:
  backward: bool = True
  optim_step: bool = False
  timeit: bool = True
  warmup_steps: int = 10
  count_steps: int = 20

@dataclass
class Config:
  mode: str # "benchmark" | "profile"
  llm: LLMConfig
  timing: TimingConfig = field(default_factory=TimingConfig)

cs = ConfigStore.instance()
cs.store(group="timing", name="forward", node=TimingConfig(backward=False, optim_step=False))
cs.store(group="timing", name="backward", node=TimingConfig(backward=True, optim_step=False))
cs.store(group="timing", name="full", node=TimingConfig(backward=True, optim_step=True))
cs.store(name="timing_nowarm", node={'timing': dict(warmup_steps=0, count_steps=10)})

cs.store(group="llm", name="small", node=LLMConfig(
  vocab_size=10000,
  batch_size=4,
  context_length=256,
  d_model=768,
  d_ff=3072,
  num_layers=12,
  num_heads=12,
))
cs.store(group="llm", name="medium", node=LLMConfig(
  vocab_size=10000,
  batch_size=4,
  context_length=256,
  d_model=1024,
  d_ff=4096,
  num_layers=24,
  num_heads=16,
))
cs.store(group="llm", name="large", node=LLMConfig(
  vocab_size=10000,
  batch_size=4,
  context_length=256,
  d_model=1280,
  d_ff=5120,
  num_layers=36,
  num_heads=20,
))
cs.store(group="llm", name="xl", node=LLMConfig(
  vocab_size=10000,
  batch_size=4,
  context_length=256,
  d_model=1600,
  d_ff=6400,
  num_layers=48,
  num_heads=25,
))
cs.store(group="llm", name="2.7B", node=LLMConfig(
  vocab_size=10000,
  batch_size=4,
  context_length=256,
  d_model=2560,
  d_ff=10240,
  num_layers=32,
  num_heads=32,
))

default_args = ["+llm=small", "+timing=full", "+mode=null"]
def get_config(args: list[str] | None = None) -> Config:
  import hydra
  from typing import cast
  if args is None:
    args = [*default_args]
  else:
    args = [*default_args, *args]

  with hydra.initialize(config_path=None, version_base='1.3'):
    cfg = hydra.compose(
      config_name=None,
      overrides=args
    )
  return cast(Config, cfg)

def get_config_multi(args: list[str] | None = None) -> list[Config]:
  import hydra
  from hydra.core.global_hydra import GlobalHydra
  from hydra.core.override_parser.overrides_parser import OverridesParser
  from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
  from typing import cast
  if args is None:
    args = [*default_args]
  else:
    args = [*default_args, *args]
  with hydra.initialize(config_path=None, version_base='1.3'):
    gh = GlobalHydra.instance()
    assert gh.hydra is not None
    parser = OverridesParser.create(config_loader=gh.hydra.config_loader)
    overrides = parser.parse_overrides(args)
    split_overrides = BasicSweeper.split_arguments(overrides, None)[0]
    configs = [
      hydra.compose(
        config_name=None,
        overrides=o # type: ignore
      ) for o in split_overrides
    ]
  return [cast(Config, cfg) for cfg in configs]

get_config(['+timing_nowarm']).timing
# get_config_multi(["+llm=small,large", "+timing=full", "llm.batch_size=16,32", "llm.context_length=128,256"])
# %%
from cs336_basics.modules import CrossEntropy
from dataclasses import dataclass, field
import timeit
import numpy as np
import torch
from torch import Tensor

def sample_xy(config: LLMConfig) -> tuple[Tensor, Tensor]:
  x = (torch.randn((config.batch_size, config.context_length + 1), device="cuda") % 1 * config.vocab_size).abs().long()
  return x[:, :-1], x[:, 1:]

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

    return {
      'forward_mean': format_unit(forward_times.mean(), unit),
      'forward_std': format_unit(forward_times.std(), unit),
      'backward_mean': format_unit(backward_times.mean(), unit) if len(backward_times) > 0 else "",
      'backward_std': format_unit(backward_times.std(), unit) if len(backward_times) > 0 else "",
      'optim_mean': format_unit(optim_times.mean(), unit) if len(optim_times) > 0 else "",
      'optim_std': format_unit(optim_times.std(), unit) if len(optim_times) > 0 else "",
      'total_mean': format_unit(total_times.mean(), unit),
      'total_std': format_unit(total_times.std(), unit),
    }

  @staticmethod
  def print_report(report: dict, unit: str = "seconds"):
    print(f"Forward: {report['forward_mean']} ± {report['forward_std']}")
    if report['backward_mean']:
      print(f"Backward: {report['backward_mean']} ± {report['backward_std']}")
    if report['optim_mean']:
      print(f"Optim: {report['optim_mean']} ± {report['optim_std']}")
    print(f"Total: {report['total_mean']} ± {report['total_std']}")

cross_entropy = CrossEntropy()
def step(llm: torch.nn.Module, optim: torch.optim.Optimizer, x: Tensor, y: Tensor, backward: bool = True, optim_step: bool = True, timeit = False) -> tuple[Tensor, Timing | None]:
  timer = Timing() if timeit else None
  with nvtx.range("forward"):
    y_hat = llm(x) # type: Tensor
    loss = cross_entropy(y_hat, y).mean() # type: Tensor
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
  return Timing.report([r for r in result if r is not None], unit="milliseconds")

def get_args():
  import sys
  args = [i for i in sys.argv[1:] if not i.startswith("--")]
  return args

if __name__ == "__main__" and (cfg := get_config(get_args())) and cfg.mode == "benchmark":
  config: LLMConfig = OmegaConf.to_object(cfg.llm) # type: ignore
  x, y = sample_xy(config)

  # %%
  llm, optim = config.create_llm(device="cuda")
  for epoch in range(cfg.timing.warmup_steps):
    step(llm, optim, x, y, backward=cfg.timing.backward, optim_step=cfg.timing.optim_step, timeit=False)
  report = timeit_steps(llm, optim, x, y, steps=cfg.timing.count_steps, backward=cfg.timing.backward, optim_step=cfg.timing.optim_step)
  Timing.print_report(report)
  # %%
  import os
  if not os.path.exists("benchmark_results.csv"):
    with open("benchmark_results.csv", "w") as f:
      f.write("d_model,num_layers,num_heads,batch_size,context_length,")
      f.write("forward_mean(ms),forward_std(ms),")
      f.write("backward_mean(ms),backward_std(ms),")
      f.write("optim_mean(ms),optim_std(ms),")
      f.write("total_mean(ms),total_std(ms)\n")
  with open("benchmark_results.csv", "a") as f:
    def get_value(s: str) -> str:
      return str(float(s.split()[0])) if s else ''
    try:
      f.write(f"{config.d_model},{config.num_layers},{config.num_heads},{config.batch_size},{config.context_length},")
      f.write(f"{get_value(report['forward_mean'])},{get_value(report['forward_std'])},")
      f.write(f"{get_value(report['backward_mean'])},{get_value(report['backward_std'])},")
      f.write(f"{get_value(report['optim_mean'])},{get_value(report['optim_std'])},")
      f.write(f"{get_value(report['total_mean'])},{get_value(report['total_std'])}")
    finally:
      f.write("\n")

  # %%
  import os
  os._exit(0)

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
if __name__ == "__main__" and (cfg := get_config(get_args())) and cfg.mode == "profile":
  cfg.mode = "profile"
  config: LLMConfig = OmegaConf.to_object(cfg.llm) # type: ignore
  x, y = sample_xy(config)
  llm, optim = config.create_llm(device="cuda")
  for epoch in range(cfg.timing.warmup_steps):
    step(llm, optim, x, y, backward=True, optim_step=True)
  for epoch in range(cfg.timing.count_steps):
    step(llm, optim, x, y, backward=cfg.timing.backward, optim_step=cfg.timing.optim_step, timeit=cfg.timing.timeit)

# %%
