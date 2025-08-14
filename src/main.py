#!/usr/bin/env python3

from pybullet import (
    disconnect,
    POSITION_CONTROL,
    setJointMotorControl2,
    getLinkState,
    changeConstraint,
    getNumJoints,
    resetJointState,
    JOINT_FIXED,
    createConstraint,
    resetBasePositionAndOrientation,
    loadSDF,
    loadURDF,
    changeVisualShape,
    connect,
    GUI,
    getAABB,
    setAdditionalSearchPath,
    setGravity,
    changeVisualShape,
    multiplyTransforms,
    stepSimulation,
    JOINT_GEAR,
    calculateInverseKinematics,
    setTimeStep,
    getBasePositionAndOrientation,
    getMatrixFromQuaternion,
    getQuaternionFromEuler,
    configureDebugVisualizer,
    COV_ENABLE_SINGLE_STEP_RENDERING,
    getJointState,
    getQuaternionFromEuler,
    calculateInverseKinematics,
    computeViewMatrix,
    computeProjectionMatrixFOV,
    getCameraImage,
    addUserDebugLine,
)
import pybullet as p
from pybullet_data import getDataPath
from threading import Thread
from scipy.constants import g
from time import sleep
from typing import Tuple
import math
from function import FunctionTable, FunctionSignatureFormatter
from math import pi
from arm import ArmController
import numpy
import numpy as np
from openai import OpenAI
import openai
from os import environ
from utility import draw_object_box
from world import WorldObject, WorldManager

import requests

####################################################################################################

TIME_STEP = 240

####################################################################################################

connect(GUI)
setAdditionalSearchPath(getDataPath())
setGravity(0, 0, -g)
setTimeStep(1 / TIME_STEP)

####################################################################################################

plane_id = loadURDF("plane.urdf")
table_id = loadURDF(
    "table/table.urdf",
    basePosition=[0.0, 0.0, 0.0],
    baseOrientation=[0, 0, 0.5, 0.5],
    globalScaling=1
)

world_manager = WorldManager()

world_manager.append(
    WorldObject(
        table_id,
        "table",
        [],
    )
)

red_id = loadURDF("cube.urdf", basePosition=[0, 0, 0.66], globalScaling=0.066)
changeVisualShape(red_id, -1, rgbaColor=[1, 0, 0, 1.0])

blue_id = loadURDF("cube.urdf", basePosition=[0, 0.5, 0.66], globalScaling=0.066)
changeVisualShape(blue_id, -1, rgbaColor=[0, 0, 1, 1.0])

green_id = loadURDF("cube.urdf", basePosition=[0, -0.5, 0.66], globalScaling=0.066)
changeVisualShape(green_id, -1, rgbaColor=[0, 1, 0, 1.0])

INITIAL_POSITIONS = {}
INITIAL_ORIENTATIONS= {}
for id in [red_id, blue_id, green_id]:
    position, orientation = getBasePositionAndOrientation(id)
    INITIAL_POSITIONS[id] = position
    INITIAL_ORIENTATIONS[id] = orientation

world_manager.append(
   WorldObject(
       red_id,
       "cube",
       ["red"],
   ),
   WorldObject(
       blue_id,
       "cube",
       ["blue"],
   ),
   WorldObject(
       green_id,
       "cube",
       ["green"],
   ),
)

arm_controller = ArmController()
client = OpenAI(
    api_key=environ.get("OPENAI_KEY"),
)

def open_gripper(context):
    context["motion_planner"].open_gripper()

def close_gripper(context):
    context["motion_planner"].close_gripper()

def move_gripper_to(
        context,
        position: Tuple[float, float, float],
        #velocity: 'float' = 2.0,
        #interpolation_type: 'None | "linear" | "smoothstep" | "cubic_spline"' = "smoothstep",
        #interpolation_steps: 'None | int' = 5,
):
    context["motion_planner"].move_gripper_to(
        position,
        #interpolation_type=interpolation_type,
        #interpolation_steps=interpolation_steps,
        velocity=2.5,
    )

def reset_position(context):
    context["motion_planner"].reset()

table = FunctionTable(FunctionSignatureFormatter())
table.register(open_gripper)
table.register(close_gripper)
table.register(move_gripper_to)
table.register(reset_position)

def reset():
    arm_controller.reset(execute_callbacks=False)
    for id in [red_id, green_id, blue_id]:
        resetBasePositionAndOrientation(id, INITIAL_POSITIONS[id], INITIAL_ORIENTATIONS[id])

def pick_red():
    # Position of the red cube
    red_cube_position = (0.0, 0.0, 0.66)
    # Move above the red cube
    move_gripper_to((red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.1))
    # Open the gripper
    open_gripper()
    # Move down to the red cube
    move_gripper_to(red_cube_position)
    # Close the gripper to pick the cube
    close_gripper()
    # Lift the cube up
    move_gripper_to((red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.1))

red_cube_position = (0.0, 0.0, 0.66)

def perform_motion(controller):
    reset()

    arm_controller.create_motion_planner().reset(
    ).move_gripper_to(
        (red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.1), linear_interpolation_steps=10,
    ).open_gripper(
    ).move_gripper_to(red_cube_position, linear_interpolation_steps=10).close_gripper().move_gripper_to(
        (red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.1), linear_interpolation_steps=10
    ).open_gripper().then(
        perform_motion
    ).apply()

#perform_motion(arm_controller)

arm_controller.create_motion_planner().reset().apply()

def receive_input():
    message = None
    intention = None#0

    while True:
        if intention is not None:
            message = INTENTIONS[intention].prompt
        else:
            message = input("> ")

            if message == "exit":
                break
            elif message == "break":
                breakpoint()
                continue
            elif message == "reset":
                reset()
                continue

        content = """
You are the controller of an robotic arm, that uses Python for interaction.
Please write and call your written Python 3 function for the command
"{message}"

Provide only the code using python``` and ```. Use only the provided functions.

{world_prompt}

Functions:
{functions_prompt}
""".format(
    message=message,
    functions_prompt=table.format_prompt_specification(),
    world_prompt=world_manager.format_prompt_specification(),
)

        print(content)

        output = ""
        model = 'gpt'

        if model == 'gpt5':
            completions = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a Python 3 code generator."},
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                temperature=1,
            )

            output = completions.choices[0].message.content.strip("\n\r ")

        if model == 'gpt':
            completions = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a Python 3 code generator."},
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                temperature=0,
            )

            output = completions.choices[0].message.content.strip("\n\r ")

        if model == 'falcon7b':
            url = "http:///qwen-2dot5-14b-instruct/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            payload = {
                "model": "falcon-h1-7b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a Python 3 code generator."},
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "temperature": 0,
            }

            response = requests.post(url, headers=headers, json=payload)
            output = response.json()["choices"][0]["message"]["content"].strip("\n\r ")

        if model == 'falcon34b':
            url = "http:///qwen-2dot5-14b-instruct/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            payload = {
                "model": "falcon-h1-34b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a Python 3 code generator."},
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "temperature": 0,
            }

            response = requests.post(url, headers=headers, json=payload)
            output = response.json()["choices"][0]["message"]["content"].strip("\n\r ")

        if model == 'phi':
            url = "http:///qwen-2dot5-14b-instruct/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            payload = {
                "model": "phi-4",
                "messages": [
                    {"role": "system", "content": "You are a Python 3 code generator."},
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "temperature": 0,
            }

            response = requests.post(url, headers=headers, json=payload)
            output = response.json()["choices"][0]["message"]["content"].strip("\n\r ")

        if model == 'qwen':
            url = "http:///qwen-2dot5-14b-instruct/v1/chat/completions"

            headers = {
                "Content-Type": "application/json",
            }

            payload = {
                "model": "qwen-2dot5-14b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a Python 3 code generator."},
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "temperature": 0,
            }

            response = requests.post(url, headers=headers, json=payload)
            output = response.json()["choices"][0]["message"]["content"].strip("\n\r ")

        marker = "```"
        left_markers = [marker + "python3", marker + "python", marker]
        right_marker = marker

        start = None
        start_padding = None

        for left_marker in left_markers:
            index = output.find(left_marker)

            if index < 0:
                continue

            start = index
            start_padding = len(left_marker)

            break

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
            return {
                "status": "failed",
                "execution": {},
                "output": output,
                "generated_code": "",
                "response_time_in_seconds": elapsed_s,
                "time_to_first_token_in_milliseconds": time_to_first_token_ms,
            }

        end = output.find(right_marker, start + start_padding)
        if end < 0:
            end = len(output)

        if end is None:
            return {
                "status": "failed",
                "execution": {},
                "output": output,
                "generated_code": "",
                "response_time_in_seconds": elapsed_s,
                "time_to_first_token_in_milliseconds": time_to_first_token_ms,
            }

        code = output[start + start_padding:end]

        print(output)

        context = {
            "motion_planner": arm_controller.create_motion_planner(),
        }

        try:
            result = table.evaluate(
                code,
                context=context,
                tracing=True,
            )

            if intention is not None:
                print("Pre Verification")

                for verifier in INTENTIONS[intention].pre_verifiers:
                    print(
                        verifier.__qualname__,
                        verifier(result["trace"], world_manager=world_manager)
                    )

            context["motion_planner"].apply()

            if intention is not None:
                print("Post Verification")

                for verifier in INTENTIONS[intention].post_verifiers:
                    print(
                        verifier.__qualname__,
                        verifier(result["trace"], world_manager=world_manager)
                    )


            intention = None
        except Exception as e:
            print("Error evaluating")
            print(e)
            reset()

thread = Thread(target=receive_input)
thread.start()

i = 0

import pybullet as p

"""
vector_id = p.addUserDebugParameter("vector", 0.0, 10.0, 0.3)
fov_id = p.addUserDebugParameter("fov", 0.0, 360.0, 120,0)
aspect_id = p.addUserDebugParameter("aspect", 0.0, 1.0, 1.0)
near_val_id = p.addUserDebugParameter("nearVal", 0, 20, 0.01)
far_val_id = p.addUserDebugParameter("farVal", 0, 100, 100)
"""

while thread.is_alive():
    if i % 60 == 0:
        """
        arm_controller.display_camera_image(
            float(p.readUserDebugParameter(vector_id)),
            float(p.readUserDebugParameter(fov_id)),
            float(p.readUserDebugParameter(aspect_id)),
            float(p.readUserDebugParameter(near_val_id)),
            float(p.readUserDebugParameter(far_val_id)),
        )
        """
        pass


    i += 1

    configureDebugVisualizer(COV_ENABLE_SINGLE_STEP_RENDERING)

    arm_controller.update(1 / TIME_STEP)

    stepSimulation()
    sleep(1 / TIME_STEP)

disconnect()
