import cv2
import mediapipe as mp

import numpy as np
from matplotlib import pyplot as plt

import time

import vector
import screen

class VirtualScreen:

    def __init__(self, max=5):
        self.frame = np.array([])
        self.max = max

        self.timer = None
        self.screen_positions = np.array([])

    def isRecognizing(self, eye, hands):
        if not (eye and hands.left and hands.right) :
            return False

        self.frame = np.append(self.frame, {'eye': eye, 'hands': hands})
        #print(len(self.frame))

        if len(self.frame) <= self.max:
            return False
        self.frame = np.delete(self.frame, 0, axis=0)
        getHandLandmark = np.vectorize(lambda frame, side, landmark_id: frame['hands'].get(side).landmark[landmark_id])

        hand = {side: {id: vector.Average3D(getHandLandmark(self.frame, side, id)) for id in [2, 4, 5, 8]} for side in ['left', 'right']}

        hand_v = {'left': vector.calcVector3D(hand['left'][2], hand['left'][4]), 'right': vector.calcVector3D(hand['right'][4], hand['right'][2])}
        hand_index = {'left': vector.calcVector3D(hand['left'][2], hand['left'][4]), 'right': vector.calcVector3D(hand['right'][2], hand['right'][4])}
        hand_thumb = {'left': vector.calcVector3D(hand['left'][5], hand['left'][8]), 'right': vector.calcVector3D(hand['right'][5], hand['right'][8])}

        index_angle = vector.calcDotProduct([hand_v['left'].x, hand_v['left'].y], [hand_v['right'].x, hand_v['right'].y])
        hand_angle = {'left': vector.calcDotProduct(hand_index['left'].parseArray(), hand_thumb['left'].parseArray()), 'right': vector.calcDotProduct(hand_index['right'].parseArray(), hand_thumb['right'].parseArray())}
        is_orthogonal = {'left': 60 < hand_angle['left'] < 110, 'right': 60 < hand_angle['right'] < 110}
        return index_angle < 45 and is_orthogonal['left'] and is_orthogonal['right']

    def calcVertex(self, eye, hands):
        isRecognising = self.isRecognizing(eye, hands)
        
        if not isRecognising:
            self.timer = None
            self.screen_positions = np.array([])
            return None, -1
        elif self.timer == None:
            self.timer = time.time()
        
        elapsed_time = time.time() - self.timer

        if elapsed_time < 1:
            return None, elapsed_time
        elif 2 < elapsed_time:
            getVector = np.vectorize(lambda frame, id: frame[id])
            screen_vertex = screen.calc_vertex(
                vector.Average3D(getVector(self.screen_positions, 'v')),
                vector.Average3D(getVector(self.screen_positions, 'left')),
                vector.Average3D(getVector(self.screen_positions, 'right')))
            self.screen_positions = np.array([])
            self.timer = None
            return screen_vertex, elapsed_time

        v = vector.calcVector3D(
            vector.calcVector3D(hands.right.landmark[5], hands.right.landmark[8]),
            vector.calcVector3D(hands.left.landmark[5], hands.left.landmark[8])
        )
        position = {'v': v, 'left': hands.left.landmark[1], 'right': hands.right.landmark[1]}
        
        self.screen_positions = np.append(self.screen_positions, position)

        #print(self.screen_positions)
        screen_vertex = screen.calc_vertex(v, position['left'], position['right'])
        return screen_vertex, elapsed_time

    def getVertex():
        return