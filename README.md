# llarm

This project investigates how Large Language Models (LLMs) can directly orchestrate robot behavior, from high-level intentions to low-level action sequences. It uses PyBullet as a reproducible simulation environment to prototype and evaluate LLM-guided policies for a robotic gripper arm, including grasping, placement, and simple manipulation routines. The pipeline translates natural-language goals into structured action plans, executes them via PyBullet, and feeds back state and failures to refine prompts, constraints, and controllers. Our objective is to map language to reliable closed-loop control, benchmarking success rates, safety, and sample efficiency entirely in simulation before transferring to hardware.

## Install

``` shell
pip install -r requirements.txt
```

## Environment Variables Overview

| Key               | Description                               | Default |
|-------------------|-------------------------------------------|---------|
| MODEL_NAME        | LLM model name                            | None    |
| MODEL_TEMPERATURE | LLM model temperature                     | 0.0     |
| MODEL_URL         | URL to the chat completions endpoint      | None    |
| MODEL_KEY         | API key for the chat completions endpoint | None    |

## Freestanding Environment

The freestanding environment is used to experiment with with different LLMs and tasks. Intentions can be typed directly into the console. Furthermore, the commands listed below can be executed for resetting, exitting or debugging the environment.

### Run

``` shell
python3 -m src/main.py
```

### Commands

| Name  | Action                                                         |
|-------|----------------------------------------------------------------|
| reset | Resets the environment, including gripper and object positions |
| exit  | Exit the application                                           |
| break | Breakpoint call to the python debugger                         |

