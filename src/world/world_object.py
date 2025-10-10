from pybullet import (
    getAABB,
    getBasePositionAndOrientation,
    getEulerFromQuaternion,
)

class WorldObject:
    """
    Represents a single object within the simulation environment.

    Each object has a unique ID assigned by PyBullet, a name, and a set of tags used for
    categorization. Additionally, it provides functions to access the object's position,
    orientation, bounding box, and dimensions by querying the simulation environment.
    """

    def __init__(self, id, name, tags):
        self.__id = id
        self.__name = name
        self.__tags = tags

    @property
    def id(self):
        return self.__id

    @property
    def name(self):
        return self.__name

    @property
    def tags(self):
        return self.__tags

    @property
    def position(self):
        return list(
            map(
                lambda x: round(x, 2),
                getBasePositionAndOrientation(self.__id)[0],
            )
        )

    @property
    def quaternion_orientation(self):
        return getBasePositionAndOrientation(self.__id)[1]

    @property
    def euler_orientation(self):
        return getEulerFromQuaternion(self.quaternion_orientation)

    @property
    def aabb(self):
        return getAABB(self.__id)

    @property
    def dimension(self):
        aabb = self.aabb

        return list(
            map(
                lambda x: round(x, 2),
                [
                    abs(aabb[0][0] - aabb[1][0]),
                    abs(aabb[0][1] - aabb[1][1]),
                    abs(aabb[0][2] - aabb[1][2]),
                ]
            )
        )

    @property
    def aabb_min(self):
        vector = list(
            map(
                lambda x: round(x, 3),
                self.aabb[0],
            )
        )

        return vector

    @property
    def aabb_max(self):
        vector = list(
            map(
                lambda x: round(x, 3),
                self.aabb[1],
            )
        )

        return vector
