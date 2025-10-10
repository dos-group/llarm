# llarm

This project investigates how Large Language Models (LLMs) can directly orchestrate robot behavior, bridging high-level intentions and low-level action sequences. PyBullet is used as a reproducible simulation environment to prototype and evaluate LLM-guided policies for a robotic gripper arm, including grasping, placement, and simple manipulation routines. The pipeline translates natural-language goals into structured action plans, executes them in PyBullet, and incorporates state feedback and failure signals to refine prompts, constraints, and controllers. The objective is to map language to reliable closed-loop control, benchmarking success rates, safety, and sample efficiency entirely in simulation before transferring to hardware. Furthermore, the project aims to provide diverse environments for investigating different aspects of workflow generation within this context.

## Research Context

Modern robotics aims not only for reliably executing low-level controls, but also for interpreting and acting upon high-level goals expressed in human language. Bridging the semantic gap between natural language instructions and low-level motor control remains a key challenge in embodied AI. While much work has focused on either (a) mapping language to discrete action sequences or (b) learning continuous control policies in robotic domains, relatively fewer systems seek to directly leverage LLMs to orchestrate robot behaviors in closed loop with perception and feedback.

## Key Components
### World

The `WorldManager` abstracts the environment in PyBullet and provides a unified interface for querying objects. It consists of instances of `WorldObject` for retrieving object related details such as the orientation or the position. It is used for providing environment related information to the prompt.

### Controller

The `ArmController` abstracts required joint movements and implements related state management. It provides high-level methods for performing gripper opening, closing and moving.

### Workflow

The `WorkflowManager` and its related classes provide the infrastructure for workflow generation, transitions, and event handling. The `WorkflowGenerator` class is responsible for interfacing with an external LLM. NodeTransformer classes, such as `WorkflowAsynchronousNodeTransformer`, implement visitors for transforming Abstract Syntax Tree (AST) representations of LLM-generated workflows. While `WorkflowFunctions` offers a high-level interface to registered functions, `WorkflowEventListeners` provides signaling and hooking mechanisms that are useful for extension, tracing, and debugging.

## Getting Started

The application requires Python 3 for execution. It provides multiple environments for different aspects of the aforementioned context. The Base environment provides a common and shared setting.

### Setup

#### virtualenv


```shell
virtualenv environment
source environment/bin/activate
pip3 install -r requirements.txt
```

#### pip

``` shell
pip install -r requirements.txt
```

### Freestanding Environment Execution

The freestanding environment is designed for experimenting with different LLMs and tasks. Intentions can be entered directly into the console from which the application was launched. The commands listed below allow resetting, exiting, or debugging the environment. Configuration of a specific LLM is handled via environment variables.

#### Environment Variables

| Key               | Description                               | Default |
|-------------------|-------------------------------------------|---------|
| MODEL_NAME        | LLM model name                            | None    |
| MODEL_TEMPERATURE | LLM model temperature                     | None    |
| MODEL_URL         | URL to the chat completions endpoint      | None    |
| MODEL_KEY         | API key for the chat completions endpoint | None    |

#### Run

``` shell
python3 -m "src.environment.freestanding"
```

#### Commands

| Name  | Action                                                                         |
|-------|--------------------------------------------------------------------------------|
| reset | Reset the environment, including gripper and object positions and joint states |
| exit  | Exit the application                                                           |
| break | Breakpoint call to the python debugger                                         |

### Reinforcement Learning Environment (TODO)

The reinforcement learning environment is intended for adapting models using reinforcement learning. Its implementation is not yet complete.

#### Environment Variables

| Key               | Description                               | Default |
|-------------------|-------------------------------------------|---------|
| MODEL_NAME        | LLM model name                            | None    |
| MODEL_TEMPERATURE | LLM model temperature                     | None    |
| MODEL_URL         | URL to the chat completions endpoint      | None    |
| MODEL_KEY         | API key for the chat completions endpoint | None    |

#### Run

``` shell
python3 -m "src.environment.reinforcement_learning"
```
