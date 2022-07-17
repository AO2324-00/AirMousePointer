import cv2
import mediapipe as mp

import numpy as np
from matplotlib import pyplot as plt

import time

import vector
import screen

class HandState:

    def __init__(self, max=5):
        self.frame = np.array([])
        self.max = max
        self.timer = None
        self.screen_positions = np.array([])
        self.screen_offsets = np.array([])

    def isRecognizing(self, hands_landmarks, pose_landmarks):
        if not (pose_landmarks.pose_landmarks and hands_landmarks.multi_hand_landmarks):
            return False

        if len(hands_landmarks.multi_hand_landmarks) < 2:
            return False

        self.frame = np.append(self.frame, {'hands': hands_landmarks.multi_hand_landmarks, 'pose': pose_landmarks.pose_landmarks})
        #print(len(self.frame))

        if len(self.frame) <= self.max:
            return False
        self.frame = np.delete(self.frame, 0, axis=0)
        getHandLandmark = np.vectorize(lambda frame, hand_id, landmark_id: frame['hands'][hand_id].landmark[landmark_id])

        hand = [{id: vector.Average3D(getHandLandmark(self.frame, i, id)) for id in [2, 4, 5, 8]} for i in range(2)]

        hand_v = [vector.calcVector3D(hand[0][2], hand[0][4]), vector.calcVector3D(hand[1][4], hand[1][2])]
        hand_index = [vector.calcVector3D(hand[0][2], hand[0][4]), vector.calcVector3D(hand[1][2], hand[1][4])]
        hand_thumb = [vector.calcVector3D(hand[0][5], hand[0][8]), vector.calcVector3D(hand[1][5], hand[1][8])]

        #index_angle = vector.calcDotProduct(hand_v[0].parseArray(), hand_v[1].parseArray())
        index_angle = vector.calcDotProduct([hand_v[0].x, hand_v[0].y], [hand_v[1].x, hand_v[1].y])
        hand_angle = [vector.calcDotProduct(hand_index[0].parseArray(), hand_thumb[0].parseArray()), vector.calcDotProduct(hand_index[1].parseArray(), hand_thumb[1].parseArray())]
        is_orthogonal = [60 < hand_angle[0] < 110, 60 < hand_angle[1] < 110]
        #print(index_angle)
        #if index_angle < 45 and is_orthogonal[0] and is_orthogonal[1]:
        if index_angle < 45 and is_orthogonal[0] and is_orthogonal[1]:
            #print(index_angle)
            #print(True, '{:6>.2f}'.format(index_angle), '{:6>.2f}'.format(hand_angle[0]), '{:6>.2f}'.format(hand_angle[1]))
            return True
        #print(False, '{:6>.2f}'.format(index_angle), '{:6>.2f}'.format(hand_angle[0]), '{:6>.2f}'.format(hand_angle[1]))
        return False

    def calcPosition(self, image, hands_landmarks, pose_landmarks):
        isRecognising = self.isRecognizing(hands_landmarks, pose_landmarks)
        
        if not isRecognising:
            self.timer = None
            self.screen_positions = np.array([])
            self.screen_offsets = np.array([])
            return None, -1, None
        elif self.timer == None:
            self.timer = time.time()
        
        elapsed_time = time.time() - self.timer

        if elapsed_time < 1:
            return None, elapsed_time, None
        elif 2 < elapsed_time:
            getVector = np.vectorize(lambda frame, id: frame[id])
            screen_vertex = screen.calc_vertex(
                vector.Average3D(getVector(self.screen_positions, 'v')),
                vector.Average3D(getVector(self.screen_positions, 'p0')),
                vector.Average3D(getVector(self.screen_positions, 'p1')))
            self.screen_positions = np.array([])
            left_offset = np.average(getVector(self.screen_offsets, 'left'))
            right_offset = np.average(getVector(self.screen_offsets, 'right'))
            self.screen_offsets = np.array([])
            self.timer = None
            return screen_vertex, elapsed_time, {'left': left_offset, 'right': right_offset}
        v0 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[0].landmark[5], hands_landmarks.multi_hand_landmarks[0].landmark[8])
        v1 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[1].landmark[5], hands_landmarks.multi_hand_landmarks[1].landmark[8])
        v = vector.Vector3D({'x': v0.x-v1.x, 'y': v0.y-v1.y, 'z': v0.z-v1.z})
        #p0 = pose_landmarks.pose_landmarks.landmark[19]
        #p1 = pose_landmarks.pose_landmarks.landmark[20]
        #p0 = pose_landmarks.pose_landmarks.landmark[15]
        #p1 = pose_landmarks.pose_landmarks.landmark[16]
        
        p0 = hands_landmarks.multi_hand_landmarks[0].landmark[1]
        p1 = hands_landmarks.multi_hand_landmarks[1].landmark[1]
        
        if calcRelativeHandsPosition(hands_landmarks, pose_landmarks):
            tmp = vector.Vector3D({'x': p0.x, 'y': p0.y, 'z': p0.z + pose_landmarks.pose_landmarks.landmark[15].z})
            p0 = vector.Vector3D({'x': p1.x, 'y': p1.y, 'z': p1.z + pose_landmarks.pose_landmarks.landmark[16].z})
            p1 = tmp
        else:
            p0 = vector.Vector3D({'x': p0.x, 'y': p0.y, 'z': p0.z + pose_landmarks.pose_landmarks.landmark[16].z})
            p1 = vector.Vector3D({'x': p1.x, 'y': p1.y, 'z': p1.z + pose_landmarks.pose_landmarks.landmark[15].z})
        
        self.screen_positions = np.append(self.screen_positions, {'v': v, 'p0': p0, 'p1': p1})

        height, width, _ = image.shape
        left_offset = vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[11], pose_landmarks.pose_landmarks.landmark[13], vector.Vector3D({'x': width, 'y': height, 'z': width}))+vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[13], pose_landmarks.pose_landmarks.landmark[15], vector.Vector3D({'x': width, 'y': height, 'z': width}))
        right_offset = vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[12], pose_landmarks.pose_landmarks.landmark[14], vector.Vector3D({'x': width, 'y': height, 'z': width}))+vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[14], pose_landmarks.pose_landmarks.landmark[16], vector.Vector3D({'x': width, 'y': height, 'z': width}))
        self.screen_offsets = np.append(self.screen_offsets, {'left': left_offset, 'right': right_offset})
        #print(self.screen_positions)
        screen_vertex = screen.calc_vertex(v, p0, p1)
        return screen_vertex, elapsed_time, {'left': left_offset, 'right': right_offset}

def calcRelativeHandsPosition(hands_landmarks, pose_landmarks):
    p0 = pose_landmarks.pose_landmarks.landmark[15]
    p1 = pose_landmarks.pose_landmarks.landmark[16]
    h0 = hands_landmarks.multi_hand_landmarks[0].landmark[0]
    h1 = hands_landmarks.multi_hand_landmarks[1].landmark[0]

    distance00 = vector.calcDistance2D(p0, h0)
    distance10 = vector.calcDistance2D(p1, h0)
    distance01 = vector.calcDistance2D(p0, h1)
    distance11 = vector.calcDistance2D(p1, h1)

    return distance00 + distance11 < distance10 + distance01