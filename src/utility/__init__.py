from pybullet import (
    getNumJoints,
    getJointState,
    getAABB,
    addUserDebugLine,
)

from .motion_planner import MotionPlanner
from .state import State


def distances_are_between(left, right):
    return map(
        lambda x: abs(x[0] - x[1]),
        zip(left, right),
    )


def every_distance_is_within_tolerance(left, right, tolerance=1e-3):
    return all(map(lambda x: x <= tolerance, distances_are_between(left, right)))


def get_joint_positions(id):
    return list(
        map(
            lambda x: getJointState(id, x)[0],
            range(getNumJoints(id)),
        )
    )

def floatify(data):
    return list(
        map(
            lambda x: float(x),
            data,
        )
    )

def draw_box(aabb):
    aabb_min, aabb_max = aabb

    f = [aabb_min[0], aabb_min[1], aabb_min[2]]
    t = [aabb_max[0], aabb_min[1], aabb_min[2]]
    addUserDebugLine(f, t, [1, 0, 0])
    f = [aabb_min[0], aabb_min[1], aabb_min[2]]
    t = [aabb_min[0], aabb_max[1], aabb_min[2]]
    addUserDebugLine(f, t, [0, 1, 0])
    f = [aabb_min[0], aabb_min[1], aabb_min[2]]
    t = [aabb_min[0], aabb_min[1], aabb_max[2]]
    addUserDebugLine(f, t, [0, 0, 1])

    f = [aabb_min[0], aabb_min[1], aabb_max[2]]
    t = [aabb_min[0], aabb_max[1], aabb_max[2]]
    addUserDebugLine(f, t, [1, 1, 1])

    f = [aabb_min[0], aabb_min[1], aabb_max[2]]
    t = [aabb_max[0], aabb_min[1], aabb_max[2]]
    addUserDebugLine(f, t, [1, 1, 1])

    f = [aabb_max[0], aabb_min[1], aabb_min[2]]
    t = [aabb_max[0], aabb_min[1], aabb_max[2]]
    addUserDebugLine(f, t, [1, 1, 1])

    f = [aabb_max[0], aabb_min[1], aabb_min[2]]
    t = [aabb_max[0], aabb_max[1], aabb_min[2]]
    addUserDebugLine(f, t, [1, 1, 1])

    f = [aabb_max[0], aabb_max[1], aabb_min[2]]
    t = [aabb_min[0], aabb_max[1], aabb_min[2]]
    addUserDebugLine(f, t, [1, 1, 1])

    f = [aabb_min[0], aabb_max[1], aabb_min[2]]
    t = [aabb_min[0], aabb_max[1], aabb_max[2]]
    addUserDebugLine(f, t, [1, 1, 1])

    f = [aabb_max[0], aabb_max[1], aabb_max[2]]
    t = [aabb_min[0], aabb_max[1], aabb_max[2]]
    addUserDebugLine(f, t, [1.0, 0.5, 0.5])
    f = [aabb_max[0], aabb_max[1], aabb_max[2]]
    t = [aabb_max[0], aabb_min[1], aabb_max[2]]
    addUserDebugLine(f, t, [1, 1, 1])
    f = [aabb_max[0], aabb_max[1], aabb_max[2]]
    t = [aabb_max[0], aabb_max[1], aabb_min[2]]
    addUserDebugLine(f, t, [1, 1, 1])

def draw_object_box(id):
    draw_box(getAABB(id))
