import numpy as np
from Commons.vector import Vector2D, Vector3D, calcVector2D, calcVector3D, calcDotProduct, calcMiddleVector, calcCornerVector
from Commons.landmarks import Landmarks, LandmarkPoint, ScreenLandmarks

def calcScreenVertex(screenLandmarks: ScreenLandmarks):
    return np.array([
        screenLandmarks.origin_point,
        LandmarkPoint(calcCornerVector(screenLandmarks.diagonal_point.landmark, screenLandmarks.origin_point.landmark, screenLandmarks.horizontal_direction.landmark), scale=screenLandmarks.scale),
        screenLandmarks.diagonal_point,
        LandmarkPoint(calcCornerVector(screenLandmarks.origin_point.landmark, screenLandmarks.diagonal_point.landmark, screenLandmarks.horizontal_direction.landmark), scale=screenLandmarks.scale)
    ])

class Plane:
    def __init__(self, screenLandmarks: ScreenLandmarks):
        self.vertex = calcScreenVertex(screenLandmarks)
        self.adjustment()
    
    def adjustment(self):
        AB = calcVector3D(self.vertex[0].landmark, self.vertex[1].landmark)
        AC = calcVector3D(self.vertex[0].landmark, self.vertex[-1].landmark)
        AB_AC = np.cross(AB.parseArray(), AC.parseArray())
        self.p = AB_AC[0]
        self.q = AB_AC[1]
        self.r = AB_AC[2]
        self.d = -(AB_AC[0]*self.vertex[0].landmark.x + AB_AC[1]*self.vertex[0].landmark.y + AB_AC[2]*self.vertex[0].landmark.z)

    def getVertex(self):
        return self.vertex

    def isIncluded(self, v):
        threshold = 1e-10
        result = self.p*v.x + self.q*v.y + self.r*v.z + self.d
        return -threshold <= result <= threshold

class VirtualScreen:
    def __init__(self, screenLandmarks: ScreenLandmarks):
        self.__screenLandmarks = screenLandmarks
        self.equation = Plane(screenLandmarks)
    
    def getScreenLandmarks(self):
        return self.__screenLandmarks

    def getVertex(self):
        return self.equation.getVertex()

    def calcIntersection(self, origin, v):
        k = -(self.equation.d + self.equation.p*origin.x + self.equation.q*origin.y + self.equation.r*origin.z)/(self.equation.p*v.x + self.equation.q*v.y + self.equation.r*v.z)
        return LandmarkPoint(Vector3D(x=v.x*k+origin.x, y=v.y*k+origin.y, z=v.z*k+origin.z), scale=self.__screenLandmarks.scale)