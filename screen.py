from matplotlib import pyplot as plt
import cv2
import numpy as np
import vector

"""
v0 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[0].landmark[5], hands_landmarks.multi_hand_landmarks[0].landmark[8])
v1 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[1].landmark[5], hands_landmarks.multi_hand_landmarks[1].landmark[8])
v = vector.Vector3D({'x': v0.x-v1.x, 'y': v0.y-v1.y, 'z': v0.z-v1.z})
"""
def calc_vertex(v, pose19, pose20):
    c0 = vector.calcCornerVector(pose19, pose20, v)
    c1 = vector.calcCornerVector(pose20, pose19, v)
    return [pose19, c0, pose20, c1]

def draw_border(image, screen_landmarks, rgb=(255, 255, 255)):
    annotated_image = image.copy()
    height, width, _ = image.shape
    for index in range(len(screen_landmarks)):
        cv2.line(annotated_image, (int(screen_landmarks[index-1].x*width), int(screen_landmarks[index-1].y*height)), (int(screen_landmarks[index].x*width), int(screen_landmarks[index].y*height)), rgb, thickness=3)
    return annotated_image

def draw_point(image, point_landmark, rgb=(255, 0, 0)):
    annotated_image = image.copy()
    height, width, _ = image.shape
    cv2.drawMarker(annotated_image, (int(point_landmark.x*width), int(point_landmark.y*height)), color=rgb, markerType=cv2.MARKER_TILTED_CROSS, thickness=2)

    return annotated_image

def plot_vertex(screen_landmarks, pose2, pose5, pose11, pose12):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([screen_landmarks[0].x, screen_landmarks[2].x], [screen_landmarks[0].y, screen_landmarks[2].y], [screen_landmarks[0].z, screen_landmarks[2].z], s=10, c="blue")

    ax.scatter([screen_landmarks[1].x, screen_landmarks[3].x], [screen_landmarks[1].y, screen_landmarks[3].y], [screen_landmarks[1].z, screen_landmarks[3].z], s=40, c="red")
    ax.plot([
        screen_landmarks[0].x, screen_landmarks[1].x, screen_landmarks[2].x, screen_landmarks[3].x, screen_landmarks[0].x],
        [screen_landmarks[0].y, screen_landmarks[1].y, screen_landmarks[2].y, screen_landmarks[3].y, screen_landmarks[0].y],
        [screen_landmarks[0].z, screen_landmarks[1].z, screen_landmarks[2].z, screen_landmarks[3].z, screen_landmarks[0].z], color='red')

    ax.plot([pose2.x, pose5.x], [pose2.y, pose5.y], [pose2.z, pose5.z], color='green') # Eye position
    ax.plot([pose11.x, pose12.x], [pose11.y, pose12.y], [pose11.z, pose12.z], color='green') # Shoulder position

    ax.set_xlim(-0, 1)
    ax.set_ylim(-0, 1)
    ax.set_zlim(-1, 0)

    plt.show()

class Plane:
    def __init__(self, screen_vertex):
        AB = vector.calcVector3D(screen_vertex[0], screen_vertex[1])
        AC = vector.calcVector3D(screen_vertex[0], screen_vertex[-1])
        AB_AC = np.cross(AB.parseArray(), AC.parseArray())
        self.p = AB_AC[0]
        self.q = AB_AC[1]
        self.r = AB_AC[2]
        self.d = -(AB_AC[0]*screen_vertex[0].x + AB_AC[1]*screen_vertex[0].y + AB_AC[2]*screen_vertex[0].z)

    def isIncluded(self, v):
        threshold = 1e-10
        result = self.p*v.x + self.q*v.y + self.r*v.z + self.d
        return -threshold <= result <= threshold

class SpatialPlane:

    def __init__(self, screen_vertex):
        self.vertex = screen_vertex
        self.equation = Plane(screen_vertex)

    def calcIntersection(self, origin, v):
        k = -(self.equation.d + self.equation.p*origin.x + self.equation.q*origin.y + self.equation.r*origin.z)/(self.equation.p*v.x + self.equation.q*v.y + self.equation.r*v.z)
        return vector.Vector3D({'x': v.x*k+origin.x, 'y': v.y*k+origin.y, 'z': v.z*k+origin.z})