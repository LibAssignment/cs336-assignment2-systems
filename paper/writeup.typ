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
