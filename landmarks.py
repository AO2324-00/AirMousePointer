import numpy as np

import vector

class BothSides:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"left: {self.left}, right: {self.right}"

    def get(self, side):
        if side == 'left':
            return self.left
        elif side == 'right':
            return self.right
        return None

    def set(self, side, value):
        if side == 'left':
            self.left = value
        elif side == 'right':
            self.right = value

class Landmarks:
    def __init__(self, eye, hands):
        self.eye = eye
        self.hands = hands

class HandLandmarks:
    def __init__(self, hand_index=range(21)):
        self.landmark = {}
        for i in hand_index:
            self.landmark[i] = vector.Vector3D({'x': 0, 'y': 0, 'z': 0})


class LandmarkParser:

    def __init__(self, offset=0.05, max=3):
        self.frame = np.array([])
        self.offset = offset
        self.max = max

    def parse(self, image, Pose=None, Hands=None, hand_index=range(21)):
        if not Pose.pose_landmarks:
            return Landmarks(None, BothSides())
        height, width, _ = image.shape
        eye_depth = vector.calcDistance3D(Pose.pose_landmarks.landmark[2], Pose.pose_landmarks.landmark[5], vector.Vector3D({'x': width, 'y': height, 'z': width}))
        eye_depth /= width
        #print(eye_depth)
        eye = vector.calcMiddleVector(Pose.pose_landmarks.landmark[2], Pose.pose_landmarks.landmark[5])
        eye.z = eye_depth
        eye.z *= 3
        eye.z -= self.offset
        #print(eye.z)

        hands = BothSides()
        if Hands.multi_hand_landmarks:
            
            if calcRelativeHandsPosition(Pose, Hands.multi_hand_landmarks[0].landmark[0]) == 'left':
                hands.left = Hands.multi_hand_landmarks[0]
                if len(Hands.multi_hand_landmarks) >= 2:
                    hands.right = Hands.multi_hand_landmarks[1]
            else:
                hands.right = Hands.multi_hand_landmarks[0]
                if len(Hands.multi_hand_landmarks) >= 2:
                    hands.left = Hands.multi_hand_landmarks[1]

            left_depth = 0
            right_depth = 0
            if hands.left:
                d0 = vector.calcDistance2D(hands.left.landmark[5], hands.left.landmark[17], vector.Vector2D({'x': width, 'y': height}))
                #d1 = vector.calcDistance2D(hands.left.landmark[2], hands.left.landmark[17], vector.Vector2D({'x': width, 'y': height}))
                #d2 = vector.calcDistance2D(hands.left.landmark[0], hands.left.landmark[5], vector.Vector2D({'x': width, 'y': height}))*0.9
                d3 = vector.calcDistance2D(hands.left.landmark[0], hands.left.landmark[17], vector.Vector2D({'x': width, 'y': height}))*0.9
                #d4 = vector.calcDistance2D(hands.left.landmark[0], hands.left.landmark[2], vector.Vector2D({'x': width, 'y': height}))
                #print('{:6>.2f}'.format(d0), '{:6>.2f}'.format(d1), '{:6>.2f}'.format(d2), '{:6>.2f}'.format(d3), '{:6>.2f}'.format(d4))
                #left_depth = max(d0, d1, d2, d3, d4)
                #print('{:6>.2f}'.format(d0), '{:6>.2f}'.format(d3))
                left_depth = max(d0, d3)
                left_depth /= width
                #print(left_depth)
                left_depth *= 3
            if hands.right:
                d0 = vector.calcDistance2D(hands.right.landmark[5], hands.right.landmark[17], vector.Vector2D({'x': width, 'y': height}))
                #d1 = vector.calcDistance2D(hands.right.landmark[2], hands.right.landmark[17], vector.Vector2D({'x': width, 'y': height}))
                #d2 = vector.calcDistance2D(hands.right.landmark[0], hands.right.landmark[5], vector.Vector2D({'x': width, 'y': height}))*0.9
                d3 = vector.calcDistance2D(hands.right.landmark[0], hands.right.landmark[17], vector.Vector2D({'x': width, 'y': height}))*0.9
                #d4 = vector.calcDistance2D(hands.right.landmark[0], hands.right.landmark[2], vector.Vector2D({'x': width, 'y': height}))
                #right_depth = max(d0, d1, d2, d3, d4)
                right_depth = max(d0, d3)
                right_depth /= width
                right_depth *= 3
            for i in hand_index:
                if hands.left: 
                    hands.left.landmark[i].z /= 100
                    hands.left.landmark[i].z += left_depth
                if hands.right:
                    hands.right.landmark[i].z /= 100
                    hands.right.landmark[i].z += right_depth

        return Landmarks(eye, hands)

    def update(self, image, Pose=None, Hands=None, hand_index=range(21)):

        result = self.parse(image, Pose, Hands, hand_index)

        if not result.eye and len(self.frame) > 0:
            self.frame = np.delete(self.frame, 0)
        else:
            self.frame = np.append(self.frame, result)
            if len(self.frame) > self.max:
                self.frame = np.delete(self.frame, 0)

        return result
    
    def get(self, hand_index=range(21)):
        result = Landmarks(vector.Vector3D({'x': 0, 'y': 0, 'z': 0}), BothSides(HandLandmarks(hand_index), HandLandmarks(hand_index)))
        eye_counter = 0
        hands_counter = BothSides(0, 0)
        for landmark in self.frame:
            if landmark.eye:
                eye_counter += 1
                result.eye = result.eye.addition(landmark.eye)
            if landmark.hands.left:
                hands_counter.left += 1
            if landmark.hands.right:
                hands_counter.right += 1
            for i in hand_index:
                if landmark.hands.left:
                    x = result.hands.left.landmark[i].x
                    result.hands.left.landmark[i] = result.hands.left.landmark[i].addition(landmark.hands.left.landmark[i])
                if landmark.hands.right:
                    result.hands.right.landmark[i] = result.hands.right.landmark[i].addition(landmark.hands.right.landmark[i])
        result.eye = result.eye.division(eye_counter)
        #print(self.frame[-1].hands.left.landmark[0].x, result.hands.left.landmark[0].x)
        for i in hand_index:
            result.hands.left.landmark[i] = result.hands.left.landmark[i].division(hands_counter.left)
            result.hands.right.landmark[i] = result.hands.right.landmark[i].division(hands_counter.right)

        return result

def calcRelativeHandsPosition(pose_landmarks, landmark):
    left = pose_landmarks.pose_landmarks.landmark[15]
    right = pose_landmarks.pose_landmarks.landmark[16]

    distance_left = vector.calcDistance2D(left, landmark)
    distance_right = vector.calcDistance2D(right, landmark)

    return 'left' if distance_left < distance_right else 'right'