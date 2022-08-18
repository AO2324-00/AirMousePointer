from typing import Union, Optional
import numpy as np
import vector
from vector import Vector3D

class BothSides:
    def __init__(self, *, left=None, right=None):
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

class HandLandmarks:
    def __init__(self, hand_index=range(21)):
        self.landmark: dict[int, Vector3D] = {}
        self.__hand_index = hand_index
        for i in self.__hand_index:
            self.landmark[i] = Vector3D(x=0, y=0, z=0)
    
    def addition(self, hand_landmarks: 'HandLandmarks') -> None:
        for i in self.__hand_index:
            self.landmark[i] = self.landmark[i].addition(hand_landmarks.landmark[i])

    def division(self, numerical: float) -> None:
        for i in self.__hand_index:
            self.landmark[i] = self.landmark[i].division(numerical)


class Landmarks:

    def __init__(self, *, eye: Optional[Vector3D], hands: Optional[BothSides], average_limit: int=20):
        self.eye = eye
        self.hands = hands if hands else BothSides()
        self.__average_limit = average_limit
        self.frames: np.ndarray = np.array([])
        self.__average_tmp: dict[int, 'Landmarks'] = {}

    def update(self, landmarks: 'Landmarks') -> 'Landmarks':
        self.frames = np.append(landmarks.frames, self)
        if self.__average_limit < len(self.frames):
            self.frames = np.delete(self.frames, 0)
        return self

    def averaged(self, frame_number: int) -> 'Landmarks':
        if frame_number in self.__average_tmp:
            return self.__average_tmp[frame_number]
        result = Landmarks(
            eye=Vector3D(x=0, y=0, z=0),
            hands=BothSides(
                left=HandLandmarks(self.frames[-1].hands.left.landmark.keys()), 
                right=HandLandmarks(self.frames[-1].hands.right.landmark.keys())))
        eye_count = 0
        hands_count = BothSides(left=0, right=0)

        frames = self.frames[len(self.frames)-frame_number:len(self.frames)]
        for landmarks in frames:
            if landmarks.eye:
                eye_count += 1
                result.eye.addition(landmarks.eye)
            for side in ['left', 'right']:
                if landmarks.hands.get(side):
                    hands_count.set(side, hands_count.get(side) + 1)
                    result.hands.get(side).addition(landmarks.hands.get(side))
    
        result.eye.division(eye_count)
        for side in ['left', 'right']:
            result.hands.get(side).division(hands_count.get(side))

        self.__average_tmp[frame_number] = result
        return result


    @staticmethod
    def parse(image: np.ndarray, *, Pose: Optional[any], Hands: Optional[any], hand_index:Union[range,list]=range(21)) -> 'Landmarks':
        if not Pose.pose_landmarks:
            return Landmarks(eye=None, hands=None)
        height, width, _ = image.shape
        eye_depth = vector.calcDistance3D(Pose.pose_landmarks.landmark[2], Pose.pose_landmarks.landmark[5], scale=Vector3D(x=width, y=height, z=width)) * 0.8
        eye_depth /= width
        eye = vector.calcMiddleVector(Vector3D.fromVector(Pose.pose_landmarks.landmark[2]), Vector3D.fromVector(Pose.pose_landmarks.landmark[5]))
        eye.z = eye_depth
        eye.z *= 10
        eye.z -= 0.05

        multi_hands = BothSides()
        hands = BothSides(left=HandLandmarks(hand_index), right=HandLandmarks(hand_index))
        if Hands.multi_hand_landmarks:
            
            if calcRelativeHandsPosition(Pose, Hands.multi_hand_landmarks[0].landmark[0]) == 'left':
                multi_hands.left = Hands.multi_hand_landmarks[0]
                if len(Hands.multi_hand_landmarks) >= 2:
                    multi_hands.right = Hands.multi_hand_landmarks[1]
            else:
                multi_hands.right = Hands.multi_hand_landmarks[0]
                if len(Hands.multi_hand_landmarks) >= 2:
                    multi_hands.left = Hands.multi_hand_landmarks[1]

            hands_depth = BothSides(left=0, right=0)
            
            for side in ['left', 'right']:
                if not multi_hands.get(side):
                    continue
                d0 = vector.calcDistance3D(multi_hands.get(side).landmark[5], multi_hands.get(side).landmark[17], scale=Vector3D(x=width, y=height, z=width))
                d3 = vector.calcDistance3D(multi_hands.get(side).landmark[0], multi_hands.get(side).landmark[17], scale=Vector3D(x=width, y=height, z=width))*0.8
                depth = max(d0, d3)
                depth /= width
                depth *= 10
                hands_depth.set(side, depth)

                for i in hand_index:
                    depth = multi_hands.get(side).landmark[i].z
                    depth /= 5
                    depth += hands_depth.get(side)
                    landmark = multi_hands.get(side).landmark[i]
                    landmark.z = depth
                    hands.get(side).landmark[i] = Vector3D.fromVector(landmark)

        return Landmarks(eye=eye, hands=hands)

class LandmarkParser:

    def __init__(self, offset=0.05, max=4):
        self.frame = np.array([])
        self.offset = offset
        self.max = max

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
        result = Landmarks(Vector3D({'x': 0, 'y': 0, 'z': 0}), BothSides(HandLandmarks(hand_index), HandLandmarks(hand_index)))
        eye_counter = 0
        hands_counter = BothSides(left=0, right=0)
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