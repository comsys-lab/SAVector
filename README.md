# Intro
This repository contains scale-out DNN accelerator simulators. The simulators are based on SCALE-Simv2.

(README is working in progress...)

## Description
`Scale_up_sim_faster.py`: Simulate the scale-up architecture with a single pod.

`Scale_out_sim_faster.py`: Simulate the scale-out architecture.

`SAVector_sim_faster.py`, `SAVector_sim_perf_mode.py`: Simulate SAVector architecture. Basicaly, we use `SAVector_sim_perf_mode.py`.

`SOSA_rev_RT.py`, `SOSA_rev_idealRT.py`: Simulate SOSA architecture. idealRT simulates the SOSA architecture with no performance degradation from shared buffer access latency.

## How to run simulation
You can run the simulation with a python code:

```python3 Scale_out_sim_faster <DNN model> <systolic array dimension> <pod dimension> <batch size>```

The above instruction working for `Scale_up_sim_faster.py`, `Scale_out_sim_faster.py`, `SAVector_sim_faster.py`, `SAVector_sim_perf_mode.py`, `SOSA_rev_RT.py`, `SOSA_rev_idealRT.py`, `Shared_buffer_sim_ideal.py`
