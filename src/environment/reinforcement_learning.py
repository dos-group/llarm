from .base import Base
from threading import Thread
from ..utility import PyBulletContext
from time import sleep
from .lift_object_rewarder import LiftObjectRewarder
from os import environ
from math import log, sqrt, acos, exp
from numpy import array

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
        self.__rewards = []
        self.__xy_delta_threshold = xy_delta_threshold
        self.__xy_delta_weight = xy_delta_weight
        self.__z_delta_weight = z_delta_weight
        self.__orientation_weight = orientation_weight
        self.__orientation_threshold = orientation_threshold

    @property
    def rewards(self):
        return self.__rewards

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

        self.__rewards.append(
            {
                position_delta_reward,
                orientation_delta_reward,
            }
        )

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

class AfterExecuteImageTracer:
    def register(self, workflow_manager):
        workflow_manager.event_listeners.after_execute.register(self.__after_execute)

    def __after_execute(self, *args, **kwargs):
        import pybullet as p
        import pybullet_data
        import numpy as np
        import cv2

        width, height = 320, 240
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1,1,1],
            cameraTargetPosition=[1,1,1],
            cameraUpVector=[0,0,1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=100.0)

        img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix)
        rgba = np.array(img_arr[2], dtype=np.uint8).reshape((height, width, 4))
        rgb = rgba[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("screenshot.png", bgr)
        print("Bild gespeichert als screenshot.png")
        
with PyBulletContext(False):
    base = Base()
    rewarder = LiftObjectRewarder(base.world_manager.query_objects('cube', ['red'])[0])
    rewarder.register(base.workflow_manager)
    AfterExecuteImageTracer().register(base.workflow_manager)

    for i in range(1):
        thread = Thread(target=lambda: base.evaluate("Pick red"))
        thread.start()

        while thread.is_alive():
            base.update()
            #sleep(1.0 / time_step)

        base.reset()

    print(rewarder.rewards)
