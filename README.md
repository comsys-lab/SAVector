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

```
python3 Scale_out_sim_faster <DNN model> <systolic array dimension> <pod dimension> <batch size>
```

For example, the code below simulate a scale-out architecture with 4x4 pod dimension and 32x32 systolic array dimension. Benchmark is DenseNet-169 with batch size=1.

```
python3 Scale_out_sim_faster DenseNet169 32x32 4x4 1
```

The above instruction working for `Scale_up_sim_faster.py`, `Scale_out_sim_faster.py`, `SAVector_sim_faster.py`, `SAVector_sim_perf_mode.py`, `SOSA_rev_RT.py`, `SOSA_rev_idealRT.py`, `Shared_buffer_sim_ideal.py`.

The simulator automatically generates the `.config` files.
To change the hardware configuration such as buffer size, you should modify the code.

