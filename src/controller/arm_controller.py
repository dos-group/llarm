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
    import numpy as np
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

def pixel_to_camera_coords(u, v, z, fx, fy, cx, cy):
    import numpy as np

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.array([X, Y, Z])

def camera_to_world(point_cam, cam_to_world):
    import numpy as np

    # Punkt homogen erweitern: [X, Y, Z, 1]
    point_cam_h = np.append(point_cam, 1.0)

    # Matrixmultiplikation
    point_world_h = cam_to_world @ point_cam_h

    # Ergebnis (x, y, z)
    return point_world_h[:3]

def quat_mul(q1, q2):
    # (x,y,z,w) Konvention wie in PyBullet
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )

class ArmController:
    STATE_IDLE = "idle"
    STATE_SLEEP = "sleep"
    STATE_RESET = "reset"
    STATE_MOVE_TO_POSITION = "move_to_position"
    STATE_MOVE_TO_OBJECT = "move_to_object"
    STATE_GRAB = "grab"
    STATE_UNGRAB = "ungrab"
    JOINT_VELOCITIES = [
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
    ]

    RESET_PLATFORM_JOINT_POSITIONS = [
        -0.000000,
        -0.000000,
        0.000000,
        1.570793,
        0.000000,
        -1.036725,
        0.000001,
    ]

    RESET_GRIPPER_BASE_POSITION = [
        0.923103,
        -0.200000,
        1.250036,
    ]

    RESET_GRIPPER_BASE_ORIENTATION = [
        -0.000000,
        0.964531,
        -0.000002,
        -0.263970,
    ]

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

    END_EFFECTOR_INDEX = 6

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

    def display_camera_image(self, vector, fov, aspect, nearVal, farVal):
        import pybullet as p
        import numpy as np

        state = getLinkState(self.__platform_id, ArmController.END_EFFECTOR_INDEX),
        position = state[0][0]
        orientation = state[0][1]
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation))

        up_vector = np.dot(rotation_matrix, [0, 1, 0])
        z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

        scale = 1
        position = [position[0] + z_axis[0] * vector,
                    position[1] + z_axis[1] * vector,
                    position[2] + z_axis[2] * vector]

        end_pos = [position[0] + z_axis[0] * scale,
                   position[1] + z_axis[1] * scale,
                   position[2] + z_axis[2] * scale]

        view_matrix = p.computeViewMatrix(position, end_pos, up_vector)

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=nearVal,
            farVal=farVal,
        )

        width, height = (768, 640)

        img_arr = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        import cv2

        rgb_array = np.reshape(img_arr[2], (height, width, 4))
        rgb_array = rgb_array[:, :, :3]
        rgb_array = rgb_array.astype(np.uint8)
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        depth_buffer = img_arr[3]
        depth = np.reshape(depth_buffer, (height, width))

        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)

        depth_meters = farVal * nearVal / (farVal - (farVal - nearVal) * depth)

        segmentation_mask = np.reshape(img_arr[4], (height, width))
        body_ids = np.unique(segmentation_mask)
        detections = []

        for body_id in body_ids:
            if body_id < 2:
                continue

            mask = (segmentation_mask == body_id)

            if not np.any(mask):
                continue

            y_coords, x_coords = np.where(mask)
            x_center = int(np.mean(x_coords))
            y_center = int(np.mean(y_coords))

            object_depths = depth_meters[mask]
            valid_depths = object_depths[np.isfinite(object_depths) & (object_depths > 0)]

            if len(valid_depths) == 0:
                distance = None
            else:
                fov_rad = np.deg2rad(fov)
                fx = fy = 0.5 * width / np.tan(fov_rad / 2)
                cx = width / 2
                cy = height / 2

                distance = float(np.median(valid_depths))

                z = depth_meters[y_center, x_center]

                point_cam = pixel_to_camera_coords(
                    x_center, y_center, z, fx, fy, cx, cy
                )

                pos, orn = p.getBasePositionAndOrientation(body_id)

                view_matrix = np.array(view_matrix).reshape(4, 4)
                cam_to_world = np.linalg.inv(view_matrix)
                point_world = camera_to_world(point_cam, cam_to_world)
                #print(f"{body_id} {point_world} real: {pos}")

        #cv2.imwrite("./depth.jpg", depth_uint8)
        #cv2.imwrite("./rgb.jpg", rgb_array)

    def reset(self, execute_callbacks=True):
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
        getattr(self, "_ArmController__update_" + self.__state.name + "_state")(delta)

    def __update_idle_state(self, delta):
        self.__transform_state(State(ArmController.STATE_IDLE))

    def __update_grab_state(self, delta):
        positions = self.__get_gripper_positions()

        condition = every_distance_is_within_tolerance(
            positions, self.__state.data["previous_positions"], tolerance=1e-3
        )
        if not condition:
            self.__state.data["previous_positions"] = positions
            return

        self.__transform_state(State(ArmController.STATE_IDLE))

    def __update_ungrab_state(self, delta):
        positions = self.__get_gripper_positions()

        condition = every_distance_is_within_tolerance(
            positions, [0, 0], tolerance=1e-3
        )
        if not condition:
            return

        self.__transform_state(State(ArmController.STATE_IDLE))

    def __update_sleep_state(self, delta):
        self.__state.data["elapsed_seconds"] += delta

        if self.__state.data["elapsed_seconds"] < self.__state_data["seconds"]:
            return

        self.__transform_state(ArmController.STATE_IDLE)

    def __update_reset_state(self, delta):
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
