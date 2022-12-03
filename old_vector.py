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
        return f"x: {self.x}, y: {self.y}, z: {self.y}"

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

def Average3D(array: ndarray):
    length = len(array)
    result = Vector3D(x=0, y=0, z=0)
    for vector in array:
        vector = Vector3D.fromVector(vector)
        result = result.addition(vector.division(length))
    return result

def Average2D(array: ndarray):
    length = len(array)
    result = Vector2D(x=0, y=0)
    for vector in array:
        vector = Vector2D.fromVector(vector)
        result = result.addition(vector.division(length))
    return result

def calcAngle(a, b, scale=Vector3D(x=1, y=1, z=1)):
    return Vector2D(
        x=np.sign(b.x - a.x) * calcDotProduct(Vector2D(x=(b.z-a.z)*scale.z, y=(b.y-a.y)*scale.y), Vector2D(x=1, y=0)),
        y=np.sign(b.y - a.y) * calcDotProduct(Vector2D(x=(b.z-a.z)*scale.z, y=(b.x-a.x)*scale.x), Vector2D(x=1, y=0))
    )

class Smoothing3D:
    def __init__(self, max, outliers=1, delay=10):
        self.outliers = outliers
        self.min = 1 + outliers*2
        self.max = self.min if max <= self.min else max
        self.vector = np.empty((0,3))
        self.delay = delay

    def update(self, vector):
        self.vector = np.append(self.vector, np.array([[vector.x, vector.y, vector.z]]), axis=0)
        if(len(self.vector) > self.max):
            self.vector = np.delete(self.vector, 0, axis=0)

    def get(self):
        if(len(self.vector) <= self.min):
            if(len(self.vector) == 0):
                return Vector3D({
                    'x': 0,
                    'y': 0,
                    'z': 0,
                })
            return Vector3D({
                'x': self.vector[-1][0],
                'y': self.vector[-1][1],
                'z': self.vector[-1][2],
            })
        basis = self.vector[-1]
        #print(basis)
        vector = np.empty((0,3))
        for i in range(len(self.vector)):
            v = np.array([[
                (self.vector[i][0] - basis[0]) * (i+self.delay) / (len(self.vector)+self.delay),
                (self.vector[i][1] - basis[1]) * (i+self.delay) / (len(self.vector)+self.delay),
                (self.vector[i][2] - basis[2]) * (i+self.delay) / (len(self.vector)+self.delay),
            ]])
            #print(v)
            vector = np.append(vector, v, axis=0)
        #print(vector)
        sort_array = np.sort(vector, axis=0)[self.outliers:-self.outliers, :] if self.outliers != 0 else vector
        average = np.average(sort_array, axis=0)
        #average = np.average(vector, axis=0)
        #print(average)
        #average = np.average(self.vector, axis=0)
        return Vector3D(
            x=average[0] + basis[0],
            y=average[1] + basis[1],
            z=average[2] + basis[2],
        )

class Smoothing2D:
    def __init__(self, max):
        self.max = max
        self.vector = np.empty((0,2))

    def update(self, vector):
        self.vector = np.append(self.vector, np.array([[vector.x, vector.y]]), axis=0)
        over_vector = int((len(self.vector) - self.max)/2+1)
        if over_vector > 0:
            for i in range(over_vector):
                self.vector = np.delete(self.vector, 0, axis=0)

    def get(self):
        if(len(self.vector) <= 0):
            return None
        average = np.average(self.vector, axis=0)
        return Vector2D(
            x=average[0],
            y=average[1],
        )

def calcCornerVector(p, o, v):
    A = np.square(v.x) + np.square(v.y) + np.square(v.z)
    B = (o.x-p.x)*v.x + (o.y-p.y)*v.y + (o.z-p.z)*v.z
    t = -B / (1e-10 if A == 0 else A)
    result = Vector3D(x=o.x+v.x*t, y=o.y+v.y*t, z=o.z+v.z*t)
    return result

def calcMiddleVector(vector_0: Union[Vector3D, Vector2D], vector_1: Union[Vector3D, Vector2D]) -> Vector3D:
    return vector_0.addition(vector_1).division(2)