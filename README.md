# llarm

This project investigates how Large Language Models (LLMs) can directly orchestrate robot behavior, from high-level intentions to low-level action sequences. It uses PyBullet as a reproducible simulation environment to prototype and evaluate LLM-guided policies for a robotic gripper arm, including grasping, placement, and simple manipulation routines. The pipeline translates natural-language goals into structured action plans, executes them via PyBullet, and feeds back state and failures to refine prompts, constraints, and controllers. Our objective is to map language to reliable closed-loop control, benchmarking success rates, safety, and sample efficiency entirely in simulation before transferring to hardware.

## Install

``` shell
pip install -r requirements.txt
```

## Run

``` shell
python3 src/main.py
```
