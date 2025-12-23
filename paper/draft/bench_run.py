# %%
!python benchmark.py mode=benchmark llm=small timing=forward timing.warmup_steps=0 timing.count_steps=10
!python benchmark.py mode=benchmark llm=small timing=forward
!python benchmark.py mode=benchmark llm=small timing=backward timing.warmup_steps=0 timing.count_steps=10
!python benchmark.py mode=benchmark llm=small timing=backward
!python benchmark.py mode=benchmark llm=small timing=full timing.warmup_steps=0 timing.count_steps=10
!python benchmark.py mode=benchmark llm=small timing=full

# %%
!python benchmark.py mode=benchmark llm=medium timing=forward timing.warmup_steps=0 timing.count_steps=10
!python benchmark.py mode=benchmark llm=medium timing=forward
!python benchmark.py mode=benchmark llm=medium timing=backward timing.warmup_steps=0 timing.count_steps=10
!python benchmark.py mode=benchmark llm=medium timing=backward
!python benchmark.py mode=benchmark llm=medium timing=full timing.warmup_steps=0 timing.count_steps=10
!python benchmark.py mode=benchmark llm=medium timing=full

# %%
