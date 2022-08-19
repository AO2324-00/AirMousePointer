from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from numpy.linalg import solve
import vector

"""
v0 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[0].landmark[5], hands_landmarks.multi_hand_landmarks[0].landmark[8])
v1 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[1].landmark[5], hands_landmarks.multi_hand_landmarks[1].landmark[8])
v = vector.Vector3D({'x': v0.x-v1.x, 'y': v0.y-v1.y, 'z': v0.z-v1.z})
"""
def calcVertex(v, p0, p1):
    c0 = vector.Vector3D(x=p0.x, y=p0.y, z=(p0.z*4 + p1.z*1)/5) 
    c2 = vector.Vector3D(x=p1.x, y=p1.y, z=(p1.z*4 + p0.z*1)/5) 
    c1 = vector.calcCornerVector(c0, c2, v)
    c3 = vector.calcCornerVector(c2, c0, v)
    screen_vertex = [p0, c1, p1, c3]
    if screen_vertex[0].y > screen_vertex[1].y:
        screen_vertex = [screen_vertex[1], screen_vertex[0], screen_vertex[3], screen_vertex[2]]
    return screen_vertex

def calcPosition(screen_vertex, point):
    vector_X = vector.calcVector3D(screen_vertex[0], screen_vertex[3])
    vector_Y = vector.calcVector3D(screen_vertex[0], screen_vertex[1])
    x = vector.calcCornerVector(screen_vertex[0], point, vector_Y)
    y = vector.calcCornerVector(screen_vertex[0], point, vector_X)
    vector_x = vector.calcVector3D(screen_vertex[0], x)
    vector_y = vector.calcVector3D(screen_vertex[0], y)
    sign_x = np.sign(vector_x.x*vector_X.x + vector_x.y*vector_X.y + vector_x.z*vector_X.z)
    sign_y = np.sign(vector_y.x*vector_Y.x + vector_y.y*vector_Y.y + vector_y.z*vector_Y.z)
    #print(sign_x, sign_y)
    return vector.Vector2D(
        x = sign_x * (vector.calcDistance3D(screen_vertex[0], x) / vector.calcDistance3D(screen_vertex[0], screen_vertex[3])),
        y = sign_y * (vector.calcDistance3D(screen_vertex[0], y) / vector.calcDistance3D(screen_vertex[0], screen_vertex[1]))
    )



def draw_border(image, screen_vertex, color=(255, 255, 255)):
    annotated_image = image.copy()
    height, width, _ = image.shape
    annotated_image = cv2.circle(annotated_image, (int(screen_vertex[0].x*width), int(screen_vertex[0].y*height)), 4, (255, 0, 0), thickness=-1)
    annotated_image = cv2.circle(annotated_image, (int(screen_vertex[1].x*width), int(screen_vertex[1].y*height)), 4, (100, 0, 0), thickness=-1)
    annotated_image = cv2.circle(annotated_image, (int(screen_vertex[2].x*width), int(screen_vertex[2].y*height)), 4, (0, 200, 0), thickness=-1)
    for index in range(len(screen_vertex)):
        cv2.line(annotated_image, (int(screen_vertex[index-1].x*width), int(screen_vertex[index-1].y*height)), (int(screen_vertex[index].x*width), int(screen_vertex[index].y*height)), color, thickness=2)
    return annotated_image

def draw_point(image, point_landmark, color=(255, 255, 255)):
    annotated_image = image.copy()
    height, width, _ = image.shape
    cv2.drawMarker(annotated_image, (int(point_landmark.x*width), int(point_landmark.y*height)), color=color, markerType=cv2.MARKER_TILTED_CROSS, thickness=2)
    return annotated_image

def draw_line(image, line_landmarks, color=(255, 255, 255)):
    annotated_image = image.copy()
    height, width, _ = image.shape
    cv2.line(annotated_image, (int(line_landmarks[0].x*width), int(line_landmarks[0].y*height)), (int(line_landmarks[1].x*width), int(line_landmarks[1].y*height)), color, thickness=2)
    return annotated_image

def plot_vertex(screen_vertex, pose2, pose5, pose11, pose12):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([screen_vertex[0].x, screen_vertex[2].x], [screen_vertex[0].y, screen_vertex[2].y], [screen_vertex[0].z, screen_vertex[2].z], s=10, c="blue")

    ax.scatter([screen_vertex[1].x, screen_vertex[3].x], [screen_vertex[1].y, screen_vertex[3].y], [screen_vertex[1].z, screen_vertex[3].z], s=40, c="red")
    ax.plot([
        screen_vertex[0].x, screen_vertex[1].x, screen_vertex[2].x, screen_vertex[3].x, screen_vertex[0].x],
        [screen_vertex[0].y, screen_vertex[1].y, screen_vertex[2].y, screen_vertex[3].y, screen_vertex[0].y],
        [screen_vertex[0].z, screen_vertex[1].z, screen_vertex[2].z, screen_vertex[3].z, screen_vertex[0].z], color='red')

    ax.plot([pose2.x, pose5.x], [pose2.y, pose5.y], [pose2.z, pose5.z], color='green') # Eye position
    ax.plot([pose11.x, pose12.x], [pose11.y, pose12.y], [pose11.z, pose12.z], color='green') # Shoulder position

    ax.set_xlim(-0, 1)
    ax.set_ylim(-0, 1)
    ax.set_zlim(-1, 0)

    plt.show()

class Plane:
    def __init__(self, screen_vertex):
        self.vertex = screen_vertex
        self.adjustment()
    
    def adjustment(self):
        AB = vector.calcVector3D(self.vertex[0], self.vertex[1])
        AC = vector.calcVector3D(self.vertex[0], self.vertex[-1])
        AB_AC = np.cross(AB.parseArray(), AC.parseArray())
        self.p = AB_AC[0]
        self.q = AB_AC[1]
        self.r = AB_AC[2]
        self.d = -(AB_AC[0]*self.vertex[0].x + AB_AC[1]*self.vertex[0].y + AB_AC[2]*self.vertex[0].z)

    def getVertex(self):
        return self.vertex

    def isIncluded(self, v):
        threshold = 1e-10
        result = self.p*v.x + self.q*v.y + self.r*v.z + self.d
        return -threshold <= result <= threshold

class SpatialPlane:

    def __init__(self, screen_vertex):
        self.equation = Plane(screen_vertex)

    def getVertex(self):
        return self.equation.getVertex()
    
    def calcIntersection(self, origin, v):
        k = -(self.equation.d + self.equation.p*origin.x + self.equation.q*origin.y + self.equation.r*origin.z)/(self.equation.p*v.x + self.equation.q*v.y + self.equation.r*v.z)
        return vector.Vector3D(x=v.x*k+origin.x, y=v.y*k+origin.y, z=v.z*k+origin.z)

class RealtimePlot:
    def __init__(self):
        self.plt = plt
        self.fig = self.plt.figure(figsize=(10, 10))
        self.ax = Axes3D(self.fig)
        self.count = 0
# xyz > zxy
    def update(self, screen_vertex, landmarks=None, target_landmarks=None):

        if screen_vertex:
            self.ax.scatter([screen_vertex[0].x, screen_vertex[2].x], [screen_vertex[0].y, screen_vertex[2].y], [screen_vertex[0].z, screen_vertex[2].z], s=10, c="blue")

            self.ax.scatter([screen_vertex[1].x, screen_vertex[3].x], [screen_vertex[1].y, screen_vertex[3].y], [screen_vertex[1].z, screen_vertex[3].z], s=40, c="red")
            self.ax.plot(
                [screen_vertex[0].x, screen_vertex[1].x, screen_vertex[2].x, screen_vertex[3].x, screen_vertex[0].x],
                [screen_vertex[0].y, screen_vertex[1].y, screen_vertex[2].y, screen_vertex[3].y, screen_vertex[0].y],
                [screen_vertex[0].z, screen_vertex[1].z, screen_vertex[2].z, screen_vertex[3].z, screen_vertex[0].z], color='red')
        if landmarks and landmarks.eye and target_landmarks:
            self.ax.scatter([landmarks.eye.x], [landmarks.eye.y], [landmarks.eye.z], s=10, c="green")

        if landmarks and landmarks.eye and target_landmarks:
            hands = lambda side, id: landmarks.hands.get(side).landmark[id]
            self.ax.scatter([target_landmarks[0].x], [target_landmarks[0].y], [target_landmarks[0].z], s=10, c="blue")
            self.ax.scatter([target_landmarks[1].x], [target_landmarks[1].y], [target_landmarks[1].z], s=10, c="red")
            self.ax.plot([landmarks.eye.x, hands('left', 5).x], [landmarks.eye.y, hands('left', 5).y], [landmarks.eye.z, hands('left', 5).z], color='blue')
            self.ax.plot([landmarks.eye.x, hands('right', 5).x], [landmarks.eye.y, hands('right', 5).y], [landmarks.eye.z, hands('right', 5).z], color='red')
        self.count += 0.5
        #self.ax.view_init(elev=0, azim=0)
        #self.ax.view_init(elev=90, azim=90)
        self.ax.set_xlabel('x') # x軸ラベル
        self.ax.set_ylabel('y') # y軸ラベル
        self.ax.set_zlabel('z') # z軸ラベル
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        self.plt.draw()
        self.plt.pause(0.001)
        self.plt.cla()
"""
    def update(self, screen, pose_landmarks, target_landmarks=None):
        if not pose_landmarks.pose_landmarks:
            return
        pose = lambda id: pose_landmarks.pose_landmarks.landmark[id]
        if screen:
            screen_vertex = screen.getVertex()
            self.ax.scatter([screen_vertex[0].x, screen_vertex[2].x], [screen_vertex[0].y, screen_vertex[2].y], [screen_vertex[0].z, screen_vertex[2].z], s=10, c="blue")

            self.ax.scatter([screen_vertex[1].x, screen_vertex[3].x], [screen_vertex[1].y, screen_vertex[3].y], [screen_vertex[1].z, screen_vertex[3].z], s=40, c="red")
            self.ax.plot(
                [screen_vertex[0].x, screen_vertex[1].x, screen_vertex[2].x, screen_vertex[3].x, screen_vertex[0].x],
                [screen_vertex[0].y, screen_vertex[1].y, screen_vertex[2].y, screen_vertex[3].y, screen_vertex[0].y],
                [screen_vertex[0].z, screen_vertex[1].z, screen_vertex[2].z, screen_vertex[3].z, screen_vertex[0].z], color='red')
        if pose_landmarks:
            self.ax.plot([pose(2).x, pose(5).x], [pose(2).y, pose(5).y], [pose(2).z, pose(5).z], color='green') # Eye position
            self.ax.plot([pose(11).x, pose(12).x], [pose(11).y, pose(12).y], [pose(11).z, pose(12).z], color='green') # Shoulder position
            self.ax.plot([pose(11).x, pose(13).x, pose(15).x, pose(19).x], [pose(11).y, pose(13).y, pose(15).y, pose(19).y], [pose(11).z, pose(13).z, pose(15).z, pose(19).z], color='green') # Left arm
            self.ax.plot([pose(12).x, pose(14).x, pose(16).x, pose(20).x], [pose(12).y, pose(14).y, pose(16).y, pose(20).y], [pose(12).z, pose(14).z, pose(16).z, pose(20).z], color='green') # Right arm
            if target_landmarks:
                self.ax.scatter([target_landmarks[0].x], [target_landmarks[0].y], [target_landmarks[0].z], s=10, c="blue")
                self.ax.scatter([target_landmarks[1].x], [target_landmarks[1].y], [target_landmarks[1].z], s=10, c="red")
                self.ax.plot([target_landmarks[0].x, pose(19).x], [target_landmarks[0].y, pose(19).y], [target_landmarks[0].z, pose(19).z], color='blue')
                self.ax.plot([target_landmarks[1].x, pose(20).x], [target_landmarks[1].y, pose(20).y], [target_landmarks[1].z, pose(20).z], color='red')
        self.count += 0.5
        #self.ax.view_init(elev=0, azim=0)
        #self.ax.view_init(elev=90, azim=90)
        self.ax.set_xlabel('x') # x軸ラベル
        self.ax.set_ylabel('y') # y軸ラベル
        self.ax.set_zlabel('z') # z軸ラベル
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(-1, 0)
        self.plt.draw()
        self.plt.pause(0.001)
        self.plt.cla()
"""
def draw_screen(positions):
    margin = 400
    height = 600
    width = 1000
    p = lambda num: int(num+margin)
    img = np.full((height+margin*2, width+margin*2, 3), 255, np.uint8)
    img = cv2.rectangle(img,(p(0),p(0)),(p(width),p(height)),(0,0,0),3)
    for position in positions:
        img = cv2.circle(img, (p(position.x*width), p(position.y*height)), 3, (0, 0, 0), thickness=-1)
    return img
