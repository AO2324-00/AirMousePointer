import numpy as np
from numpy import linalg as LA

class Vector2D:
    def __init__(self, vector):
        self.x = vector["x"]
        self.y = vector["y"]

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}"
    
    def get(self, dir):
        if dir == 'x':
            return self.x
        elif dir == 'y':
            return self.y
        else:
            return None

    def set(self, dir, value):
        if dir == 'x':
            self.x = value
        elif dir == 'y':
            self.y = value
        
    def parseArray(self):
        return np.array([self.x, self.y])
    
    def addition(self, v):
        return Vector2D({
            'x': self.x + v.x,
            'y': self.y + v.y,
        })

    def multiply(self, num):
        return Vector2D({
            'x': self.x * num,
            'y': self.y * num,
        })

    def division(self, num):
        num = 0.00001 if num == 0 else num
        return Vector2D({
            'x': self.x / num,
            'y': self.y / num,
        })

    def equals(self, vector):
        return self.x == vector.x and self.y == vector.y

class Vector3D:
    def __init__(self, vector):
        self.x = vector["x"]
        self.y = vector["y"]
        self.z = vector["z"]

    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.y}"

    def get(self, dir):
        if dir == 'x':
            return self.x
        elif dir == 'y':
            return self.y
        elif dir == 'z':
            return self.z
        else:
            return None

    def set(self, dir, value):
        if dir == 'x':
            self.x = value
        elif dir == 'y':
            self.y = value
        elif dir == 'z':
            self.z = value


    def parseArray(self):
        return np.array([self.x, self.y, self.z])
    
    def addition(self, v):
        return Vector3D({
            'x': self.x + v.x,
            'y': self.y + v.y,
            'z': self.z + v.z,
        })
    
    def multiply(self, num):
        return Vector3D({
            'x': self.x * num,
            'y': self.y * num,
            'z': self.z * num,
        })

    def division(self, num):
        num = 0.00001 if num == 0 else num
        return Vector3D({
            'x': self.x / num,
            'y': self.y / num,
            'z': self.z / num,
        })

    def equals(self, vector):
        return self.x == vector.x and self.y == vector.y and self.z == vector.z

"""
origin=Vector3D({'x': 0, 'y': 0, 'z': 0})
point=Vector3D({'x': 0, 'y': 0, 'z': 1})
print(calcAngle(origin, point))
"""

def calcDistance3D(o, p=Vector3D({'x': 0, 'y': 0, 'z': 0}), scale=Vector3D({'x': 1, 'y': 1, 'z': 1})):
    return np.linalg.norm(np.array([o.x*scale.x, o.y*scale.y, o.z*scale.z])-np.array([p.x*scale.x, p.y*scale.y, p.z*scale.z]))

def calcDistance2D(o, p=Vector2D({'x': 0, 'y': 0}), scale=Vector3D({'x': 1, 'y': 1, 'z': 1})):
    return np.linalg.norm(np.array([o.x*scale.x, o.y*scale.y])-np.array([p.x*scale.x, p.y*scale.y]))

def calcVector3D(o, p):
    return Vector3D({
        'x': p.x - o.x,
        'y': p.y - o.y,
        'z': p.z - o.z,
    })
def calcVector2D(o, p):
    return Vector2D({
        'x': p.x - o.x,
        'y': p.y - o.y,
    })

def calcDotProduct(u=np.array([0, 0]), v=np.array([1, 0])):
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    if(n == 0):
        n = 0.01
    c = i / n
    deg = np.rad2deg(np.arccos(c))
    return deg

def calcAngle(a, b, scale=Vector3D({'x': 1, 'y': 1, 'z': 1})):
    return Vector2D({
        'x': np.sign(b.x - a.x) * calcDotProduct(np.array([(b.z-a.z)*scale.z, (b.y-a.y)*scale.y])),
        'y': np.sign(b.y - a.y) * calcDotProduct(np.array([(b.z-a.z)*scale.z, (b.x-a.x)*scale.x]))
    })


def Average3D(array):
    x = 0
    y = 0
    z = 0
    for vector in array:
        x += vector.x/len(array)
        y += vector.y/len(array)
        z += vector.z/len(array)
    return Vector3D({
        'x': x,
        'y': y,
        'z': z
    })

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
        return Vector3D({
            'x': average[0] + basis[0],
            'y': average[1] + basis[1],
            'z': average[2] + basis[2],
        })

class Smoothing2D:
    def __init__(self, max):
        self.max = max
        self.vector = np.empty((0,2))

    def update(self, vector):
        self.vector = np.append(self.vector, np.array([[vector.x, vector.y]]), axis=0)
        over_vector = int((len(self.vector) - self.max)/2+1)
        if over_vector > 0:
            print(over_vector)
            for i in range(over_vector):
                self.vector = np.delete(self.vector, 0, axis=0)

    def get(self):
        if(len(self.vector) <= 0):
            return None
        average = np.average(self.vector, axis=0)
        return Vector2D({
            'x': average[0],
            'y': average[1],
        })

def calcCornerVector(p, o, v):
    A = np.square(v.x) + np.square(v.y) + np.square(v.z)
    B = (o.x-p.x)*v.x + (o.y-p.y)*v.y + (o.z-p.z)*v.z
    t = -B / A
    result = Vector3D({'x': o.x+v.x*t, 'y':  o.y+v.y*t, 'z':  o.z+v.z*t})
    return result

def calcMiddleVector(o, p):
    return Vector3D({'x': (o.x+p.x)/2, 'y':  (o.y+p.y)/2, 'z':  (o.z+p.z)/2})

def NormalizedLandmarkToVector3D(landmark):
    return Vector3D({'x': landmark.x, 'y':  landmark.y, 'z':  landmark.z})