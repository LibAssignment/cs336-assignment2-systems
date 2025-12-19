# %%
import math
import cs336_basics2.model
from cs336_basics2.model import BasicsTransformerLM
from cs336_basics2.nn_utils import cross_entropy
from cs336_basics2.optimizer import AdamW


import timeit
from pydantic import BaseModel
import torch
from torch import Tensor
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool, Int
from einops import einsum

# %%


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
  d_k = K.shape[-1]
  with nvtx.range("computing attention scores"):
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

  if mask is not None:
    attention_scores = torch.where(mask, attention_scores, float("-inf"))

  with nvtx.range("computing softmax"):
    attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

  with nvtx.range("final matmul"):
    output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
  return output


cs336_basics2.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

# %%


class Config(BaseModel):
  vocab_size: int
  context_length: int
  d_model: int
  num_layers: int
  num_heads: int
  d_ff: int
  rope_theta: float

  batch_size: int

  def create_model_and_optimizer(self, device: str = "cuda") -> tuple[BasicsTransformerLM, AdamW]:
    model = BasicsTransformerLM(
        vocab_size=self.vocab_size,
        context_length=self.context_length,
        d_model=self.d_model,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        d_ff=self.d_ff,
        rope_theta=self.rope_theta,
    )
    model.to(device)
    optimizer = AdamW(model.parameters())
    return model, optimizer


def run_benchmark(config: Config, warmup_steps: int = 5, measurement_steps: int = 10, include_backward: bool = True):
  model, optimizer = config.create_model_and_optimizer(device="cuda")
  x = torch.randint(0, config.vocab_size, (config.batch_size,
                                           config.context_length), device='cuda')
  y = torch.randint(0, config.vocab_size, (config.batch_size,
                                           config.context_length), device='cuda')

  def model_run():
    torch.cuda.synchronize()
    forward_start = timeit.default_timer()
    y_pred = model(x)
    torch.cuda.synchronize()
    forward_end = timeit.default_timer()
    forward_cost = forward_end - forward_start

    if not include_backward:
      return forward_cost, 0.0

    loss = cross_entropy(y_pred, y)
    torch.cuda.synchronize()
    backward_start = timeit.default_timer()
    loss.backward()
    torch.cuda.synchronize()
    backward_end = timeit.default_timer()
    optimizer.step()
    optimizer.zero_grad()
    backward_cost = backward_end - backward_start
    return forward_cost, backward_cost

  forward_costs = []
  backward_costs = []
  for warmup_step in range(warmup_steps):
    # print(f"Warmup step {warmup_step + 1}/{warmup_steps}")
    model_run()
  for measurement_step in range(measurement_steps):
    # print(f"Measurement step {measurement_step + 1}/{measurement_steps}")
    forward_cost, backward_cost = model_run()
    forward_costs.append(forward_cost)
    backward_costs.append(backward_cost)
  return forward_costs, backward_costs


# %%
small_config = Config(
    vocab_size=10_000,
    context_length=256,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0,
    batch_size=4,
)
medium_config = Config(
    vocab_size=10_000,
    context_length=128,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    d_ff=4096,
    rope_theta=10000.0,
    batch_size=4,
)
large_config = Config(
    vocab_size=10_000,
    context_length=128,
    d_model=1280,
    num_layers=36,
    num_heads=20,
    d_ff=5120,
    rope_theta=10000.0,
    batch_size=4,
)
xl_config = Config(
    vocab_size=10_000,
    context_length=128,
    d_model=1600,
    num_layers=48,
    num_heads=25,
    d_ff=6400,
    rope_theta=10000.0,
    batch_size=4,
)
_2_7B_config = Config(
    vocab_size=10_000,
    context_length=128,
    d_model=2560,
    num_layers=32,
    num_heads=32,
    d_ff=10240,
    rope_theta=10000.0,
    batch_size=4,
)


# %%
from torch.version import cuda
# %%
config = Config(
    vocab_size=10_000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000.0,
    batch_size=32,
)
forward_costs, backward_costs = run_benchmark(config, 5, 10)

# %%
import numpy as np
forward_costs = np.array(forward_costs)
backward_costs = np.array(backward_costs)
print(f"Forward: mean {forward_costs.mean():.4f}s, std {forward_costs.std():.4f}s")
print(f"Backward: mean {backward_costs.mean():.4f}s, std {backward_costs.std():.4f}s")
# %%
