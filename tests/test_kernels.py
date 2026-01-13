import pytest
import torch
from torch import Tensor
from .adapters import get_softmax_functions


@pytest.mark.parametrize("input_shape", [
    (10,),
    (1024,),
    (5, 1024),
    (2, 3, 1024),
    (4, 5, 6, 1024),
])
def test_get_softmax_functions(input_shape: tuple[int, ...]):
  functions = get_softmax_functions()

  input_tensor = torch.randn(*input_shape)
  reference_output = torch.softmax(input_tensor, dim=-1)

  if functions.torch is not None:
    torch_output = functions.torch().apply(input_tensor)
    torch.testing.assert_close(torch_output, reference_output, rtol=1e-5, atol=1e-5)

  if functions.torch_tile is not None:
    torch_tile_output = functions.torch_tile().apply(input_tensor)
    torch.testing.assert_close(torch_tile_output, reference_output, rtol=1e-5, atol=1e-5)

  if functions.triton is not None:
    triton_output = functions.triton().apply(input_tensor)
    torch.testing.assert_close(triton_output, reference_output, rtol=1e-5, atol=1e-5)

  if functions.tilelang is not None:
    tilelang_output = functions.tilelang().apply(input_tensor)
    torch.testing.assert_close(tilelang_output, reference_output, rtol=1e-5, atol=1e-5)
