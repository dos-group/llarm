from pybullet import (
    disconnect,
    getAABB,
    POSITION_CONTROL,
    VELOCITY_CONTROL,
    setJointMotorControl2,
    getLinkState,
    changeConstraint,
    getNumJoints,
    multiplyTransforms,
    resetJointState,
    JOINT_FIXED,
    createConstraint,
    resetBasePositionAndOrientation,
    loadSDF,
    invertTransform,
    loadURDF,
    connect,
    GUI,
    setAdditionalSearchPath,
    setGravity,
    changeVisualShape,
    stepSimulation,
    JOINT_GEAR,
    calculateInverseKinematics,
    calculateInverseKinematics2,
    setTimeStep,
    getBasePositionAndOrientation,
    getQuaternionFromEuler,
    configureDebugVisualizer,
    getMatrixFromQuaternion,
    COV_ENABLE_SINGLE_STEP_RENDERING,
    getJointState,
    addUserDebugLine,
    addUserDebugText,
)
from math import pi
from ..utility import (
    every_distance_is_within_tolerance,
    distances_are_between,
    get_joint_positions,
    State,
    MotionPlanner,
    floatify,
)
from numpy import array, linspace, sqrt, linalg
from scipy.interpolate import CubicSpline

def interpolate_linear(start, end, steps):
    return [start + (end - start) * t for t in linspace(0, 1, steps)]

def interpolate_smoothstep(start, end, steps):
    ts = linspace(0, 1, steps)
    smooth_ts = 3 * ts**2 - 2 * ts**3
    return [start + (end - start) * t for t in smooth_ts]

def interpolate_cubic_spline(start, end, steps):
    ts = [0, 1]

    spline_x = CubicSpline(ts, [start[0], end[0]])
    spline_y = CubicSpline(ts, [start[1], end[1]])
    spline_z = CubicSpline(ts, [start[2], end[2]])

    return [
        np.array([spline_x(t), spline_y(t), spline_z(t)]) for t in linspace(0, 1, steps)
    ]

    return path

class ArmController:
    """
    Low-level controller for a robotic gripper arm.

    The `ArmController` encapsulates simulation-related routines for manipulating joint positions.
    It provides a high-level interface, such as `close_gripper()`, `open_gripper()`, and
    `move_gripper_to()`, which handle the necessary control of joint motors, joint position
    transformations, and state machine transitions.

    The state machine coordinates a sequence of actions, such as lifting an object, which requires
    moving the gripper to a position, closing the gripper, and moving it to an alternative position.
    Each high-level method sets the joint motors of the robotic gripper arm to target positions, which
    resembles the particular action, mimicking the low-level interface of real-world robotic systems.

    However, these target joint positions are not reached instantly. They are achieved gradually
    as the simulation progresses via the `stepSimulation()` function of PyBullet. Therefore, it is
    necessary to frequently check whether the target joint positions have been reached to allow
    the state machine to transition to the next state.

    This check is performed by the `update()` method, which must be called regularly within the
    simulation loop. Depending on the current state, it verifies whether the joint motor positions
    have reached their targets and advances the state machine accordingly.
    """

    STATE_IDLE = "idle"
    STATE_SLEEP = "sleep"
    STATE_RESET = "reset"
    STATE_MOVE_TO_OBJECT = "move_to_object"
    STATE_GRAB = "grab"
    STATE_UNGRAB = "ungrab"

    """
    Velocities for each joint motor.
    """
    JOINT_VELOCITIES = [
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
    ]

    """
    Fixed joint positions for the reset pose of the platform.
    """
    RESET_PLATFORM_JOINT_POSITIONS = [
        -0.000000,
        -0.000000,
        0.000000,
        1.570793,
        0.000000,
        -1.036725,
        0.000001,
    ]

    """
    Fixed joint positions for the reset pose of the gripper.
    """
    RESET_GRIPPER_JOINT_POSITIONS = [
        0.000000,
        -0.011130,
        -0.206421,
        0.205143,
        -0.009999,
        0.000000,
        -0.010055,
        0.000000,
    ]

    """
    Index of the endeffector of the gripper.
    """
    END_EFFECTOR_INDEX = 6

    """
    Platform to gripper offset
    """
    FRAME_POSITION = [0, 0, -0.025]

    def __init__(self):
        self.__platform_id = loadURDF(
            "kuka_iiwa/model_vr_limits.urdf",
            0.5,
            0,
            0.625,
            0,
            0,
            0,
            1,
        )
        self.__gripper_id = loadSDF(
            "gripper/wsg50_one_motor_gripper_new_free_base.sdf"
        )[0]
        self.__state = State(ArmController.STATE_IDLE)
        self.__states = []

        createConstraint(
            parentBodyUniqueId=self.__platform_id,
            parentLinkIndex=6,
            childBodyUniqueId=self.__gripper_id,
            childLinkIndex=0,
            jointType=JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=ArmController.FRAME_POSITION,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=getQuaternionFromEuler([0, 0, pi]),
        )
        constraint_id = createConstraint(
            self.__gripper_id,
            4,
            self.__gripper_id,
            6,
            jointType=JOINT_GEAR,
            jointAxis=[1, 1, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        changeConstraint(
            constraint_id, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100
        )

        self.reset()

    @property
    def state(self):
        return self.__state

    @property
    def platform_id(self):
        return self.__platform_id

    @property
    def states(self):
        return self.__states

    @property
    def gripper_id(self):
        return self.__gripper_id

    def create_motion_planner(self):
        return MotionPlanner(self)

    def sleep(self, seconds):
        return self.__transform_state(
            State(
                ArmController.STATE_SLEEP,
                {
                    "seconds": seconds,
                    "elapsed_seconds": 0.0,
                },
            )
        )

    def reset(self, execute_callbacks=True):
        """
        Transitions the controller into the reset state, positioning all joints
        into their predefined reset pose.
        """
        for index in range(getNumJoints(self.__platform_id)):
            setJointMotorControl2(
                self.__platform_id,
                index,
                POSITION_CONTROL,
                ArmController.RESET_PLATFORM_JOINT_POSITIONS[index],
                0,
            )

        for index in range(getNumJoints(self.__gripper_id)):
            setJointMotorControl2(
                self.__gripper_id,
                index,
                POSITION_CONTROL,
                ArmController.RESET_GRIPPER_JOINT_POSITIONS[index],
                0,
            )

        return self.__transform_state(
            State(ArmController.STATE_RESET),
            execute_callbacks=execute_callbacks,
        )

    def close_gripper(self, force=250, position=0.05):
        """
        Transitions the controller into the close-gripper state, moving the gripper
        joints to their predefined closed pose for object grasping.
        """
        for index in [4, 6]:
            setJointMotorControl2(
                bodyIndex=self.__gripper_id,
                jointIndex=index,
                controlMode=POSITION_CONTROL,
                targetPosition=position,
                force=force,
            )

        return self.__transform_state(
            State(
                ArmController.STATE_GRAB,
                {
                    "position": position,
                    "previous_positions": self.__get_gripper_positions(),
                },
            )
        )

    def idle(self, execute_callbacks=True):
        """
        Transitions the controller into the idle state, halting the operation of
        all joint motors immediately.
        """
        joint_positions = get_joint_positions(self.__platform_id)

        for index in range(getNumJoints(self.__platform_id)):
            setJointMotorControl2(
                bodyUniqueId=self.__platform_id,
                jointIndex=index,
                controlMode=VELOCITY_CONTROL,
                targetPosition=joint_positions[index],
                maxVelocity=0,
            )

        return self.__transform_state(
            State(ArmController.STATE_IDLE),
            execute_callbacks=execute_callbacks,
        )

    def open_gripper(self, force=250):
        """
        Transitions the controller into the open-gripper state, moving the gripper
        joints to their predefined open pose.
        """
        for index in [4, 6]:
            setJointMotorControl2(
                bodyIndex=self.__gripper_id,
                jointIndex=index,
                controlMode=POSITION_CONTROL,
                targetPosition=0,
                force=force,
            )

        return self.__transform_state(State(ArmController.STATE_UNGRAB))

    def move_gripper_to(
            self,
            position,
            velocity = 1.5,
            interpolation_type = None,
            interpolation_steps = None,
            ticking_handler=None,
            ticking_divisor=None,
    ):
        """
        Transitions the controller into the move-to-object state, positioning the
        gripper joints at the specified coordinates.

        The motion supports multiple interpolation modes, including linear, smoothstep,
        and cubic spline interpolation for experimentation and fine control.
        """
        if interpolation_steps is None:
            interpolation_steps = 0

        start = array(self.__get_endeffector_position())
        end = array(floatify(position))

        steps = []

        if interpolation_type == "linear":
            steps = interpolate_linear(start, end, interpolation_steps)
        elif interpolation_type == "smoothstep":
            steps = interpolate_smoothstep(start, end, interpolation_steps)
        elif interpolation_type == "cubic_spline":
            steps = interpolate_cubic_spline(start, end, interpolation_steps)

        return self.__transform_state(
            State(
                ArmController.STATE_MOVE_TO_OBJECT,
                {
                    "steps": [
                        #*steps,
                        end,
                    ],
                    "velocity": velocity,
                    "ticking_handler": ticking_handler,
                    "ticking_divisor": ticking_divisor,
                    "current_tick": 0,
                    "position": position,
                },
            )
        )

    def update(self, delta):
        """
        Updates the current state of the controller, progressing state transitions
        as needed.
        """
        getattr(self, "_ArmController__update_" + self.__state.name + "_state")(delta)

    def __update_idle_state(self, delta):
        self.__transform_state(State(ArmController.STATE_IDLE))

    def __update_grab_state(self, delta):
        """
        Checks if the conditions of the grab state are satisfied, allowing
        progression to the next state in the state machine.
        """
        positions = self.__get_gripper_positions()

        condition = every_distance_is_within_tolerance(
            positions, self.__state.data["previous_positions"], tolerance=1e-3
        )
        if not condition:
            self.__state.data["previous_positions"] = positions
            return

        self.__transform_state(State(ArmController.STATE_IDLE))

    def __update_ungrab_state(self, delta):
        """
        Checks if the conditions of the ungrab state are satisfied, allowing
        progression to the next state in the state machine.
        """
        positions = self.__get_gripper_positions()

        condition = every_distance_is_within_tolerance(
            positions, [0, 0], tolerance=1e-3
        )
        if not condition:
            return

        self.__transform_state(State(ArmController.STATE_IDLE))

    def __update_sleep_state(self, delta):
        """
        Checks if the conditions of the sleep state are satisfied, allowing
        progression to the next state in the state machine.
        """
        self.__state.data["elapsed_seconds"] += delta

        if self.__state.data["elapsed_seconds"] < self.__state_data["seconds"]:
            return

        self.__transform_state(ArmController.STATE_IDLE)

    def __update_reset_state(self, delta):
        """
        Checks if the conditions of the reset state are satisfied, allowing
        progression to the next state in the state machine.
        """
        platform_condition = every_distance_is_within_tolerance(
            get_joint_positions(self.__platform_id),
            ArmController.RESET_PLATFORM_JOINT_POSITIONS,
        )

        gripper_condition = every_distance_is_within_tolerance(
            get_joint_positions(self.__gripper_id),
            ArmController.RESET_GRIPPER_JOINT_POSITIONS,
        )

        if not (platform_condition and gripper_condition):
            return

        self.__transform_state(State(ArmController.STATE_IDLE))

    TCP_OFFSET = [0.0, 0.0, 0.25]

    def __update_move_to_object_state(self, delta):
        """
        Checks if the conditions of the move-to-object state are satisfied, allowing
        progression to the next state in the state machine.

        Since this state involves trajectory computation, each `step` in the sequence
        represents a coordinate that the gripper arm must move to. The joint poses
        are calculated using inverse kinematics. Upon reaching the target position
        of a step, the next step is taken into account. Once the final step is reached,
        the state transitions to the next state in the state machine.
        """
        if "current_step" not in self.__state.data or self.__state.data["current_step"] is None:
            self.__state.data["current_step"] = 0

        if "joint_poses" not in self.__state.data or self.__state.data["joint_poses"] is None:
            orientation = getQuaternionFromEuler([0.0, 1.00 * pi, 0.0])

            position, _ = multiplyTransforms(
                self.__state.data["steps"][self.__state.data["current_step"]],
                orientation,
                array(ArmController.FRAME_POSITION) + array([0, 0, -0.285]),
                orientation,
            )

            self.__state.data["joint_poses"] = calculateInverseKinematics(
                self.__platform_id,
                6,
                position,
                targetOrientation=getQuaternionFromEuler([0, pi, 0]),
                maxNumIterations=100,
                residualThreshold=1e-4,
            )
            for index in range(getNumJoints(self.__platform_id)):
                setJointMotorControl2(
                    self.__platform_id,
                    index,
                    POSITION_CONTROL,
                    targetPosition=self.__state.data["joint_poses"][index],
                    maxVelocity=self.__state.data["velocity"],
                )

        platform_condition = every_distance_is_within_tolerance(
            get_joint_positions(self.__platform_id),
            self.__state.data["joint_poses"],
            tolerance=1e-2,
        )

        self.__handle_ticking()

        if not platform_condition:
            return

        self.__state.data["current_step"] += 1
        del self.__state.data["joint_poses"]

        if self.__state.data["current_step"] < len(self.__state.data["steps"]):
            return

        self.__transform_state(State(ArmController.STATE_IDLE))

    def __handle_ticking(self):
        """
        Optionally, a state can define a ticking handler to directly manage
        simulation-related routines at each step.

        The tick rate is determined by the current tick and the ticking divisor.
        """
        if "current_tick" not in self.__state.data or self.__state.data["current_tick"] is None:
            return

        self.__state.data["current_tick"] += 1

        if "ticking_handler" not in self.__state.data or self.__state.data["ticking_handler"] is None:
            return

        if "ticking_divisor" not in self.__state.data or self.__state.data["ticking_divisor"] is None:
            return

        if self.__state.data["current_tick"] % self.__state.data["ticking_divisor"] == 0:
            self.__state.data["ticking_handler"]()

    def __transform_state(self, next_state, execute_callbacks=True):   
        """
        Transitions the controller to `next_state`, executing any associated
        `then_callbacks` if they exist.
        """
        previous_state = self.__state
        self.__state = next_state

        if execute_callbacks and len(previous_state.then_callbacks) > 0:
            (previous_state.then_callbacks.pop(0))(self)

            self.__state.then_callbacks = previous_state.then_callbacks + self.__state.then_callbacks

        if previous_state.name != next_state.name:
            self.__states.append(previous_state)

        return next_state

    def __get_gripper_positions(self):
        return list(
            map(
                lambda x: getJointState(self.__gripper_id, x)[0],
                [4, 6],
            ),
        )

    def __get_endeffector_position(self):
        return multiplyTransforms(
            getLinkState(
                self.__platform_id,
                ArmController.END_EFFECTOR_INDEX,
                computeForwardKinematics=True,
            )[4],
            getQuaternionFromEuler([0.0, 0, 0.0]),
            ArmController.TCP_OFFSET,
            [0.0, 0.0, 0.0, 1.0],
        )[0]
        
    def get_endeffector_position(self):
        return multiplyTransforms(
            getLinkState(
                self.__platform_id,
                ArmController.END_EFFECTOR_INDEX,
                computeForwardKinematics=True,
            )[4],
            getQuaternionFromEuler([0.0, 0, 0.0]),
            ArmController.TCP_OFFSET,
            [0.0, 0.0, 0.0, 1.0],
        )[0]
