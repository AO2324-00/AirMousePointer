from typing import Union
import numpy as np
from numpy import linalg as LA, ndarray

class Vector2D:

    @staticmethod
    def fromArray(array: Union[ndarray, list]) -> 'Vector2D':
        return Vector2D(x=array[0], y=array[1])

    @staticmethod
    def fromVector(vector) -> 'Vector2D':
        return Vector2D(x=vector.x, y=vector.y)

    @staticmethod
    def fromDict(dict: dict) -> 'Vector2D':
        return Vector2D(x=dict['x'], y=dict['y'])

    def __init__(self, *, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}"

    def get(self, dir: str) -> float:
        if dir == 'x':
            return self.x
        elif dir == 'y':
            return self.y
        else:
            return None

    def set(self, dir: str, value: float) -> None:
        if dir == 'x':
            self.x = value
        elif dir == 'y':
            self.y = value
        return self


    def parseArray(self) -> ndarray:
        return np.array([self.x, self.y])
    
    def addition(self, vector: 'Vector2D') -> 'Vector2D':
        return Vector2D(
            x = self.x + vector.x,
            y = self.y + vector.y,
        )

    def subtraction(self, vector: 'Vector2D') -> 'Vector2D':
        return Vector2D(
            x = self.x - vector.x,
            y = self.y - vector.y,
        )
    
    def multiply(self, numerical: float) -> 'Vector2D':
        return Vector2D(
            x = self.x * numerical,
            y = self.y * numerical,
        )

    def division(self, numerical: float) -> 'Vector2D':
        numerical = 1e-10 if numerical == 0 else numerical
        return Vector2D(
            x = self.x / numerical,
            y = self.y / numerical,
        )

    def equals(self, vector: 'Vector2D') -> 'Vector2D':
        return self.x == vector.x and self.y == vector.y

class Vector3D:

    @staticmethod
    def fromArray(array: ndarray or list) -> 'Vector3D':
        return Vector3D(x=array[0], y=array[1], z=array[2])

    @staticmethod
    def fromVector(vector) -> 'Vector3D':
        return Vector3D(x=vector.x, y=vector.y, z=vector.z)

    @staticmethod
    def fromDict(dict: dict) -> 'Vector3D':
        return Vector3D(x=dict['x'], y=dict['y'], z=dict['z'])

    def __init__(self, *, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"

    def get(self, dir: str) -> float:
        if dir == 'x':
            return self.x
        elif dir == 'y':
            return self.y
        elif dir == 'z':
            return self.z
        else:
            return None

    def set(self, dir: str, value: float) -> None:
        if dir == 'x':
            self.x = value
        elif dir == 'y':
            self.y = value
        elif dir == 'z':
            self.z = value
        return self


    def parseArray(self) -> ndarray:
        return np.array([self.x, self.y, self.z])
    
    def addition(self, vector: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            x = self.x + vector.x,
            y = self.y + vector.y,
            z = self.z + vector.z,
        )

    def subtraction(self, vector: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            x = self.x - vector.x,
            y = self.y - vector.y,
            z = self.z - vector.z,
        )
    
    def multiply(self, numerical: float) -> 'Vector3D':
        return Vector3D(
            x = self.x * numerical,
            y = self.y * numerical,
            z = self.z * numerical,
        )

    def division(self, numerical: float) -> 'Vector3D':
        numerical = 1e-10 if numerical == 0 else numerical
        return Vector3D(
            x = self.x / numerical,
            y = self.y / numerical,
            z = self.z / numerical,
        )

    def equals(self, vector: 'Vector3D') -> bool:
        return self.x == vector.x and self.y == vector.y and self.z == vector.z

def calcDistance3D(vector_0: Vector3D, vector_1: Vector3D=Vector3D(x=0, y=0, z=0), *, scale: Vector3D=Vector3D(x=1, y=1, z=1)) -> float:
    return np.linalg.norm(np.array([vector_0.x*scale.x, vector_0.y*scale.y, vector_0.z*scale.z])-np.array([vector_1.x*scale.x, vector_1.y*scale.y, vector_1.z*scale.z]))

def calcDistance2D(vector_0: Vector2D, vector_1: Vector2D=Vector2D(x=0, y=0), *, scale: Vector2D=Vector2D(x=1, y=1)) -> float:
    return np.linalg.norm(np.array([vector_0.x*scale.x, vector_0.y*scale.y])-np.array([vector_1.x*scale.x, vector_1.y*scale.y]))

def calcVector3D(origin_vector: Vector3D, target_vector: Vector3D) -> Vector3D:
    return target_vector.subtraction(origin_vector)

def calcVector2D(origin_vector: Vector2D, target_vector: Vector2D) -> Vector2D:
    return target_vector.subtraction(origin_vector)

def calcDotProduct(vector_0: Union[Vector3D, Vector2D], vector_1: Union[Vector3D, Vector2D]) -> float:
    vector_0 = vector_0.parseArray()
    vector_1 = vector_1.parseArray()
    inner = np.inner(vector_0, vector_1)
    norm = LA.norm(vector_0) * LA.norm(vector_1)
    c = inner / (norm if norm != 0 else 0.01)
    deg = np.rad2deg(np.arccos(c))
    return deg

def calcCornerVector(p: Vector3D, o: Vector3D, v: Vector3D) -> Vector3D:
    A = np.square(v.x) + np.square(v.y) + np.square(v.z)
    B = (o.x-p.x)*v.x + (o.y-p.y)*v.y + (o.z-p.z)*v.z
    t = -B / (1e-10 if A == 0 else A)
    result = Vector3D(x=o.x+v.x*t, y=o.y+v.y*t, z=o.z+v.z*t)
    return result

def calcMiddleVector(vector_0: Union[Vector3D, Vector2D], vector_1: Union[Vector3D, Vector2D]) -> Vector3D:
    return vector_0.addition(vector_1).division(2)