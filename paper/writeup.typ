== Problem (benchmarking_script)

+ forward
  ```python
  %timeit -n10 step(x, y, backward=False) # type: ignore
  %timeit -n20 step(x, y, backward=False) # type: ignore
  ```
  - w/o warmup: 121 ms ± 33.5 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
  - warmup: 98.9 ms ± 2.49 ms per loop (mean ± std. dev. of 7 runs, 20 loops each)
+ forward + backward
  ```python
  %timeit -n10 step(x, y, backward=True) # type: ignore
  %timeit -n20 step(x, y, backward=True) # type: ignore
  ```
  - w/o warmup: 347 ms ± 50.5 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
  - warmup: 327 ms ± 7.34 ms per loop (mean ± std. dev. of 7 runs, 20 loops each)

== Problem (nsys_profile)
+ find a bug that `torch.arange` does not set device, so it would alloc and sync to GPU, causing python code blocking and waiting for GPU to finish the alloc/sync operation.
+ there might be command queue in GPU, so launching kernel might be blocking on CPU if the queue is full.
+ if CUDA HW doesn't appear in the profile, it might be `CuptiUseRawGpuTimestamps` issue, see https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536/8 for more information.
