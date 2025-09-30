from .base import Base
from threading import Thread, Event
from ..utility import PyBulletContext
from ..workflow.core import WorkflowFunctionSignatureFormatter
from time import sleep
from os import environ
from math import log, sqrt, acos, exp
from numpy import array
from asyncio import run
from pybullet import saveState, restoreState

time_step = int(environ.get("TIME_STEP", "240"))

class LiftObjectRewarder:
    def __init__(
            self,
            object,
            xy_delta_threshold=0.01,
            xy_delta_weight=1,
            z_delta_weight=1,
            orientation_weight=1,
            orientation_threshold=0.01,
    ):
        self.__object = object
        self.__reward = None
        self.__xy_delta_threshold = xy_delta_threshold
        self.__xy_delta_weight = xy_delta_weight
        self.__z_delta_weight = z_delta_weight
        self.__orientation_weight = orientation_weight
        self.__orientation_threshold = orientation_threshold

    @property
    def reward(self):
        return self.__reward

    def register(self, workflow_manager):
        workflow_manager.event_listeners.before_execute.register(self.__before_execute)
        workflow_manager.event_listeners.after_execute.register(self.__after_execute)

    def __before_execute(self, *args, **kwargs):
        self.__position = self.__object.position
        self.__orientation = self.__object.quaternion_orientation

    def __after_execute(self, *args, **kwargs):
        position_delta_reward = self.__reward_position_delta(
            self.__object.position,
            self.__position,
        )
        orientation_delta_reward = self.__reward_orientation_delta(
            self.__object.quaternion_orientation,
            self.__orientation,
        )
        total_reward = position_delta_reward + orientation_delta_reward

        self.__reward = {
            "position_delta_reward": position_delta_reward,
            "orientation_delta_reward": orientation_delta_reward
        }

    def __reward_position_delta(self, position_from, position_to):
        position_delta = array(position_from) - position_to

        xy_delta_reward = 0
        if self.__xy_delta_threshold >= sqrt(position_delta[0] ** 2 + position_delta[1] ** 2):
            xy_delta_reward = 1

        z_delta_reward = sqrt(position_delta[2])

        return self.__z_delta_weight * z_delta_reward + self.__xy_delta_weight * xy_delta_reward

    def __reward_orientation_delta(self, orientation_from, orientation_to):
        dot = min(
            1.0,
            max(
                -1,0,
                abs(
                    orientation_from[0] * orientation_to[0] +
                    orientation_from[1] * orientation_to[1] +
                    orientation_from[2] * orientation_to[2] +
                    orientation_from[3] * orientation_to[3]
                )
            )
        )

        angle = 2.0 * acos(dot)

        score = exp(-angle / self.__orientation_threshold)

        return self.__orientation_weight * score

def formatting_prompts_func(base, example):
    return [
        {
            "role": "system",
            #"content": "You are a Python 3 code generator, you have the following met",
            "content": """
You are a Python 3 code generator. You are the controller of an robotic arm, that uses Python for interaction.
Please write and call your written Python 3 function with the name main for the command. 
Provide the code using python``` and ```.

Output:
python```
def main():
    # Call functions here

main()
```

Use only the provided functions to fulfill the user's intention.
Functions:
{functions_prompt}""".format(
                functions_prompt="\n".join(
                    map(
                        WorkflowFunctionSignatureFormatter().format,
                        [*base.workflow_manager.functions.functions],
                    )
                ),
            )
        },
        {
            "role": "user",
            "content": """
World:
{world_prompt}

Intention:
{example}""".format(
                world_prompt=base.world_manager.format_prompt_specification(),
                example=example,
            )
        }
    ]

def get_prompts(base):
    return list(
        map(
            lambda prompt: {"prompt": formatting_prompts_func(base, prompt), "method_args": {"color": "red"}},
            ["Pick red", "Pick red", "Pick red", "Pick red", "Pick red","Pick red", "Pick red", "Pick red", "Pick red", "Pick red"],
        )
    )

def extract_with_marker(output):
    marker = "```"

    left_markers = [marker + "python3", marker + "python", marker]
    right_marker = marker

    output = output.strip("\n\r ")

    start = None
    start_padding = None

    for left_marker in left_markers:
        index = output.find(left_marker)

        if index < 0:
            continue

        start = index
        start_padding = len(left_marker)

        break

    if start is None:
        return None

    end = output.find(right_marker, start + start_padding)
    if end < 0:
        end = len(output)

    if end is None:
        return None

    return output[start + start_padding:end]




def calculate_reward(state, base, llm_completion, **kwargs):
    restoreState(state)
    
    object_color = kwargs["color"]
    rewarder = LiftObjectRewarder(
        base.world_manager.query_objects("cube", [object_color])[0],
    )
    rewarder.register(base.workflow_manager)

    llm_completion = extract_with_marker(llm_completion)


    #print("[CODE] " + str(llm_completion))

    # No code generated
    if llm_completion is None:
        return -1.0
    
    event = Event()
    
    def evaluate():
        try:
            execution = run(base.workflow_manager.execute(llm_completion, timeout_in_seconds=5))
        except Exception as e:
            print(e)
        finally:
            event.set()

    
    def update():
        while not event.is_set():
            base.update()
    
    evaluate_thread = Thread(target=evaluate)
    evaluate_thread.start()
    
    update_thread = Thread(target=update)
    update_thread.start()

    evaluate_thread.join()
    update_thread.join()

    # Something went wrong?
    if rewarder.reward is None:
        return -1.0

    #print("position_delta_reward: " + str(rewarder.reward["position_delta_reward"]))
    #print("orientation_delta_reward: " + str(rewarder.reward["orientation_delta_reward"]))

    return rewarder.reward["position_delta_reward"] + rewarder.reward["orientation_delta_reward"]
