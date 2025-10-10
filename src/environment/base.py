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
    removeBody,
)
from pybullet_data import getDataPath
from scipy.constants import g
from ..utility import any_lambda
from typing import Tuple
from os import environ
from time import sleep
from ..controller import ArmController
from ..utility import point_in_aabb, draw_object_box, every_distance_is_within_tolerance
from ..world import WorldObject, WorldManager
from ..workflow.core import WorkflowGenerator, WorkflowFunctionSignatureFormatter, WorkflowManager, WorkflowEventListener
from ..workflow.utility import WorkflowExecutionTracer, WorkflowExecutionPrinter
from asyncio import get_running_loop, run
from traceback import print_exc

class Base:
    """
    Base environment for robotic gripper arm experiments.

    This environment provides the foundational setup for experiments involving a robotic gripper
    arm and a set of colored cubes. It integrates various simulation components to support
    interactive and autonomous manipulation tasks.

    The `WorkflowManager` serves as the central component for generating and executing workflows,
    which may be produced, for example, by a Large Language Model (LLM). It is combined with the
    `ArmController`, which offers a high-level interface for controlling the robotic gripper arm
    (e.g., opening, closing, and moving). Together, these components enable the simulated robot
    to be controlled programmatically or via natural-language-driven workflows.

    Executing both the simulation and the workflow concurrently (while considering Python’s
    Global Interpreter Lock, GIL) requires two threads:
    1. **Simulation thread:** Handles the simulation loop. Each step in the simulation is
    performed by the `update()` method, which calls PyBullet’s `stepSimulation()` function and
    the `update()` method of the `ArmController`.
    2. **Workflow thread:** Executes the actual workflow logic, invoking the high-level methods
    of the `ArmController`.

    The coordination between both threads is handled by the `open_gripper()`, `close_gripper()`,
    `move_gripper_to()`, and `reset_position()` methods of the `Base` class. From the perspective
    of the LLM, the executed workflow appears synchronous. However, scenarios involving external
    events reveal its inherently asynchronous nature.

    For example, when lifting an object, the robotic gripper arm must sequentially open the gripper,
    move it to the object's location, close the gripper, and then lift it vertically. These actions
    must occur in strict order. Therefore, each call to the high-level methods
    `open_gripper()`, `close_gripper()`, `move_gripper_to()`, and `reset_position()` must only
    return once the corresponding action has been successfully completed.

    Because PyBullet operates on a step-based simulation model, this coordination requires an
    internal notification or event system. This is implemented using Python’s coroutines and
    futures, allowing asynchronous task management. Consequently, the methods
    `open_gripper()`, `close_gripper()`, `move_gripper_to()`, and `reset_position()` of the
    `Base` class are defined as asynchronous. Their execution completes upon a relevant state
    transition within the `ArmController`.
    """

    def __init__(self, objects = None, sleep = True, time_step = 240):
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

        self.__table_id = table_id
        self.__world_manager = WorldManager()
        self.__object_ids = []
        self.__sleep = sleep
        
        if objects is None:
            objects = [
                {
                    "position": [0, 0, 0.66],
                    "size": 0.07,
                    "color": "red",
                },
                {
                    "position": [0, 0.5, 0.66],
                    "size": 0.07,
                    "color": "blue",
                },
                {
                    "position": [0, -0.5, 0.66],
                    "size": 0.07,
                    "color": "green",
                },
            ]

        self.__arm_controller = ArmController()
        
        self.reload_objects(objects)

        self.__workflow_manager = WorkflowManager()
        self.__workflow_manager.event_listeners.create('on_update')
        self.__workflow_manager.functions.register(self.open_gripper)
        self.__workflow_manager.functions.register(self.close_gripper)
        self.__workflow_manager.functions.register(self.move_gripper_to)
        self.__workflow_manager.functions.register(self.reset_position)

    def reload_objects(self, objects):
        for object_id in self.__object_ids:
            removeBody(object_id)

        for object_id in [self.__arm_controller.gripper_id, self.__arm_controller.platform_id]:
            removeBody(object_id)

        self.__arm_controller = ArmController()
            
        self.__object_ids = []
        self.__initial_positions = {}
        self.__initial_orientations = {}
        self.__world_manager.clear()
        
        self.__world_manager.append(WorldObject(self.__table_id, "table", []))
        
        colors = {
            "red": [1, 0, 0, 1.0],
            "blue": [0, 0, 1, 1.0],
            "green": [0, 1, 0, 1.0],
        }
        
        for object in objects:
            id = loadURDF("cube.urdf", basePosition=object['position'], globalScaling=object['size'])
            changeVisualShape(id, -1, rgbaColor=colors[object['color']])

            for link in range(getNumJoints(id)):
                changeDynamics(id, link, lateralFriction=50.0, friction_anchor=True)

            self.__object_ids.append(id)
            self.__world_manager.append(WorldObject(id, "cube", [object['color']]))

            position, orientation = getBasePositionAndOrientation(id)
            self.__initial_positions[id] = position
            self.__initial_orientations[id] = orientation

    def create_overview_image(self, height=240, width=320):
        from pybullet import computeViewMatrix, computeProjectionMatrixFOV, getCameraImage, stepSimulation, ER_TINY_RENDERER
        from numpy import reshape, array, uint8
        
        image = getCameraImage(
            width,
            height,
            computeViewMatrix(
                cameraEyePosition=[-1.5, -1.5, 1.5],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1],
            ),
            computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.1,
                farVal=100,
            ),
            renderer=ER_TINY_RENDERER,
            shadow=0,
        )
        
        rgb_array = array(image[2], dtype=uint8).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
        
    def create_front_image(self, height=240, width=320):
        from pybullet import computeViewMatrix, computeProjectionMatrixFOV, getCameraImage, stepSimulation, ER_TINY_RENDERER
        from numpy import reshape, array, uint8
        
        image = getCameraImage(
            width,
            height,
            computeViewMatrix(
                cameraEyePosition=[-1.5, 0, 1.33],
                cameraTargetPosition=[0, 0, 0.33],
                cameraUpVector=[0, 0, 1],
            ),
            computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.1,
                farVal=100,
            ),
            renderer=ER_TINY_RENDERER,
            shadow=0,
        )
        
        rgb_array = array(image[2], dtype=uint8).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def create_top_image(self, height=240, width=320):
        from pybullet import computeViewMatrix, computeProjectionMatrixFOV, getCameraImage, stepSimulation, ER_TINY_RENDERER
        from numpy import reshape, array, uint8

        image = getCameraImage(
            width,
            height,
            computeViewMatrix(
                cameraEyePosition=[0, 0, 2],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[-1, 0, 0],
            ),
            computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.1,
                farVal=100,
            ),
            renderer=ER_TINY_RENDERER,
            shadow=0,
        )

        rgb_array = array(image[2], dtype=uint8).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
        
    
    def show_image(self, image, height=240, width=320):
        import matplotlib.pyplot as plt 

        plt.imshow(image)
        plt.axis("off")
        plt.show()

    @property
    def workflow_manager(self):
        return self.__workflow_manager

    @property
    def arm_controller(self):
        return self.__arm_controller

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
        """
        Evaluates the given user intention using the `WorkflowGenerator` to produce
        output from a Large Language Model (LLM), which is then executed by the
        `WorkflowManager`.
        """
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

        if self.__sleep is True:
            sleep(1 / self.__time_step)

    async def open_gripper(self):
        """
        A function registered for access by the Large Language Model (LLM).
        It executes the corresponding workflow step and interacts with state
        callbacks to enable asynchronous execution.
        """
        loop = get_running_loop()
        future = loop.create_future()

        def handle(*args, **kwargs):
            self.__arm_controller.open_gripper().then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))

        self.__arm_controller.state.then(handle)

        await future

    async def close_gripper(self):
        """
        A function registered for access by the Large Language Model (LLM).
        It executes the corresponding workflow step and interacts with state
        callbacks to enable asynchronous execution.
        """
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
        """
        A function registered for access by the Large Language Model (LLM).
        It executes the corresponding workflow step and interacts with state
        callbacks to enable asynchronous execution.
        """
        loop = get_running_loop()
        future = loop.create_future()

        #print("move", position)

        def handle(*args, **kwargs):
            self.__arm_controller.move_gripper_to(
                position,
                velocity=1.5,
            ).then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))


        self.__arm_controller.state.then(handle)

        await future

    async def reset_position(self):
        """
        A function registered for access by the Large Language Model (LLM).
        It executes the corresponding workflow step and interacts with state
        callbacks to enable asynchronous execution.
        """
        loop = get_running_loop()
        future = loop.create_future()

        def handle(*args, **kwargs):
            self.__arm_controller.reset().then(any_lambda(lambda:loop.call_soon_threadsafe(future.set_result, True)))

        self.__arm_controller.state.then(handle)

        await future

    def reset(self):
        self.__arm_controller.reset(execute_callbacks=False)
        for id in self.__object_ids:
            resetBasePositionAndOrientation(id, self.__initial_positions[id], self.__initial_orientations[id])
