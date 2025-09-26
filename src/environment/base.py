from pybullet import (
    loadURDF,
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
    changeDynamics,
    getDynamicsInfo,
)
from pybullet_data import getDataPath
from scipy.constants import g
from ..utility import any_lambda
from typing import Tuple
from os import environ
from ..controller import ArmController
from ..utility import point_in_aabb, draw_object_box, every_distance_is_within_tolerance
from ..world import WorldObject, WorldManager
from ..workflow.core import WorkflowGenerator, WorkflowFunctionSignatureFormatter, WorkflowManager, WorkflowEventListener
from ..workflow.utility import WorkflowExecutionTracer, WorkflowExecutionPrinter
from asyncio import get_running_loop, run
from traceback import print_exc

class Base:
    def __init__(self, time_step = 240):
        self.__time_step = time_step

        setAdditionalSearchPath(getDataPath())
        setGravity(0, 0, -g)
        setTimeStep(1 / time_step)
        configureDebugVisualizer(COV_ENABLE_SINGLE_STEP_RENDERING, 0)

        plane_id = loadURDF("plane.urdf")
        table_id = loadURDF(
            "table/table.urdf",
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0.5, 0.5],
            globalScaling=1
        )

        self.__world_manager = WorldManager()

        self.__world_manager.append(
            WorldObject(
                table_id,
                "table",
                [],
            )
        )

        self.__red_id = loadURDF("cube.urdf", basePosition=[0, 0, 0.66], globalScaling=0.066)
        changeVisualShape(self.__red_id, -1, rgbaColor=[1, 0, 0, 1.0])

        self.__blue_id = loadURDF("cube.urdf", basePosition=[0, 0.5, 0.66], globalScaling=0.066)
        changeVisualShape(self.__blue_id, -1, rgbaColor=[0, 0, 1, 1.0])

        self.__green_id = loadURDF("cube.urdf", basePosition=[0, -0.5, 0.66], globalScaling=0.066)
        changeVisualShape(self.__green_id, -1, rgbaColor=[0, 1, 0, 1.0])

        self.__initial_positions = {}
        self.__initial_orientations = {}
        for id in [self.__red_id, self.__blue_id, self.__green_id]:
            position, orientation = getBasePositionAndOrientation(id)
            self.__initial_positions[id] = position
            self.__initial_orientations[id] = orientation

            for link in range(getNumJoints(id)):
                changeDynamics(id, link, lateralFriction=50.0, friction_anchor=True)

        self.__world_manager.append(
            WorldObject(
                self.__red_id,
                "cube",
                ["red"],
            ),
            WorldObject(
                self.__blue_id,
                "cube",
                ["blue"],
            ),
            WorldObject(
                self.__green_id,
                "cube",
                ["green"],
            ),
        )

        self.__arm_controller = ArmController()

        self.__workflow_manager = WorkflowManager()
        self.__workflow_manager.event_listeners.create('on_update')
        self.__workflow_manager.functions.register(self.open_gripper)
        self.__workflow_manager.functions.register(self.close_gripper)
        self.__workflow_manager.functions.register(self.move_gripper_to)
        self.__workflow_manager.functions.register(self.reset_position)

    @property
    def workflow_manager(self):
        return self.__workflow_manager

    @property
    def world_manager(self):
        return self.__world_manager

    def create_prompt(self, message):
        return """
You are the controller of an robotic arm, that uses Python for interaction.
Please write and call your written Python 3 function for the command
"{message}"

Provide only the code using python``` and ```. Use only the provided functions.

{world_prompt}

Functions:
{functions_prompt}
""".format(
    message=message,
    functions_prompt="\n".join(
       map(
           WorkflowFunctionSignatureFormatter().format,
           [*self.__workflow_manager.functions.functions],
       )
    ),
    world_prompt=self.__world_manager.format_prompt_specification(),
)

    def evaluate(self, message):
        content = self.create_prompt(message)

        print(content)

        context = {}

        generator = WorkflowGenerator(
            prompt=content,
            model_name=environ.get("MODEL_NAME", ''),
            model_url = environ.get("MODEL_URL", None),
            model_key = environ.get("MODEL_KEY", None),
            model_temperature = environ.get("MODEL_TEMPERATURE", None),
            azure_client=bool(environ.get("AZURE_CLIENT", "false")),
        )

        try:
            result = run(
                self.__workflow_manager.execute(
                    generator.generate(),
                    context=context,
                )
            )
        except Exception as e:
            print("Error evaluating")
            print(e)
            print_exc()


    def update(self):
        run(self.__workflow_manager.event_listeners.on_update.trigger())

        self.__arm_controller.update(1 / self.__time_step)
        stepSimulation()

        from time import sleep
        sleep(1 / self.__time_step)

    async def open_gripper(self):
        loop = get_running_loop()
        future = loop.create_future()

        def handle(*args, **kwargs):
            self.__arm_controller.open_gripper().then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))

        self.__arm_controller.state.then(handle)

        await future

    async def close_gripper(self):
        loop = get_running_loop()
        future = loop.create_future()

        def handle(*args, **kwargs):
            self.__arm_controller.close_gripper().then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))

        self.__arm_controller.state.then(handle)

        await future

    async def move_gripper_to(
            self,
            position: Tuple[float, float, float],
            #velocity: 'float' = 2.0,
            #interpolation_type: 'None | "linear" | "smoothstep" | "cubic_spline"' = "smoothstep",
            #interpolation_steps: 'None | int' = 5,
    ):
        loop = get_running_loop()
        future = loop.create_future()

        def handle(*args, **kwargs):
            self.__arm_controller.move_gripper_to(
                position,
                velocity=1.5,
            ).then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))


        self.__arm_controller.state.then(handle)

        await future

    async def reset_position(self):
        loop = get_running_loop()
        future = loop.create_future()

        def handle(*args, **kwargs):
            self.__arm_controller.reset().then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))

        self.__arm_controller.state.then(handle)

        await future

    def reset(self):
        self.__arm_controller.reset(execute_callbacks=False)
        for id in [self.__red_id, self.__green_id, self.__blue_id]:
            resetBasePositionAndOrientation(id, self.__initial_positions[id], self.__initial_orientations[id])
