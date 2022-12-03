from .vector import Vector2D, Vector3D, calcDistance2D, calcVector2D, calcMiddleVector
from . import kalman
import numpy as np


class Landmark(Vector3D):

    def __init__(self, landmark, *, max_track=50, smooth=True):
        super(Landmark, self).__init__(x=landmark.x, y=landmark.y, z=landmark.z)
        self.max_track = max_track
        self.smooth = smooth
        self.tracks = np.array([Vector3D(x=self.x, y=self.y, z=self.z)])
        self.__smooth_vector = None

    def update(self, landmark):
        self.tracks = np.append(self.tracks, Vector3D(x=landmark.x, y=landmark.y, z=landmark.z))
        if len(self.tracks) > self.max_track:
           self.tracks = np.delete(self.tracks, 0)
        landmark = self.get()
        self.x = landmark.x
        self.y = landmark.y
        self.z = landmark.z
        self.__smooth_vector = None

    def get(self):
        if not self.smooth:
            return Vector3D(x=self.x, y=self.y, z=self.z)
        if len(self.tracks) <= 2:
            return Vector3D(x=self.x, y=self.y, z=self.z)
        result = Vector3D(x=0, y=0, z=0)
        if not self.__smooth_vector:
            for dir in ["x", "y", "z"]:
                state = np.array([track.get(dir) for track in self.tracks])
                #state, _ = kalman.calcState(state, state[0], 0, 1000, 5000)
                state, _ = kalman.calcState(state, state[0], 0, 4000, 6000)
                #print(type(state))
                #state, _ = kalman.calcSmoothState(state, _)
                result.set(dir, state[-1])
            self.__smooth_vector = result
        return self.__smooth_vector

def getRawLandmark(landmark: Landmark, scale: Vector2D) -> Vector3D:
    _landmark = landmark.get()
    return Vector3D(x=_landmark.x*scale.x, y=_landmark.y*scale.y, z=_landmark.z*scale.x)

class Pose:

    def __init__(self, mediapipe_pose, *, scale: Vector2D=None):
        self.landmark: dict[int, Landmark] = {}
        self.raw_landmark: dict[int, Landmark] = {}
        self.scale = scale
        self.__index = range(33)
        for i in self.__index:
            self.landmark[i] = Landmark(mediapipe_pose.pose_landmarks.landmark[i])
            if self.scale:
                self.raw_landmark[i] = getRawLandmark(self.landmark[i], self.scale)
    
    def update(self, mediapipe_pose, *, scale: Vector2D=None):
        if not mediapipe_pose or not mediapipe_pose.pose_landmarks:
            return None
        for i in self.__index:
            self.landmark[i].update(mediapipe_pose.pose_landmarks.landmark[i])
            if scale:
                self.scale = scale
            if self.scale:
                self.raw_landmark[i] = getRawLandmark(self.landmark[i], self.scale)
        return self

class Hand:
    
    def __init__(self, mediapipe_hands, *, scale: Vector2D=None):
        self.landmark: dict[int, Landmark] = {}
        self.raw_landmark: dict[int, Landmark] = {}
        self.scale = scale
        self.__index = range(21)
        for i in self.__index:
            self.landmark[i] = Landmark(mediapipe_hands.landmark[i])
            if self.scale:
                self.raw_landmark[i] = getRawLandmark(self.landmark[i], self.scale)
    
    def update(self, mediapipe_hands, *, scale: Vector2D=None):
        if not mediapipe_hands:
            return None
        if scale:
            self.scale = scale
        for i in self.__index:
            self.landmark[i].update(mediapipe_hands.landmark[i])
            if self.scale:
                self.raw_landmark[i] = getRawLandmark(self.landmark[i], self.scale)
        return self


class Hands:
    def __init__(self, *, left=None, right=None, scale: Vector2D=None):
        self.scale = scale
        self.left = self.__left = Hand(left, scale=self.scale) if left else None
        self.right = self.__right = Hand(right, scale=self.scale) if right else None

    def update(self, *, left=None, right=None, scale: Vector2D=None):
        if scale:
            self.scale = scale
        if self.__left:
            self.left = self.__left.update(left, scale=self.scale)
        elif left:
            self.left = self.__left = Hand(left, scale=self.scale)
        else:
            self.left = self.__left = None
        if self.__right:
            self.right = self.__right.update(right, scale=self.scale)
        elif right:
            self.right = self.__right = Hand(right, scale=self.scale)
        else:
            self.right = self.__right = None
        return self

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

def calcRelativeHandsPosition(mediapipe_pose, mediapipe_hands):
    left = None
    right = None
    distance_left = 1
    distance_right = 1
    if not mediapipe_hands.multi_hand_landmarks or not mediapipe_pose.pose_landmarks:
        return {"left": left, "right": right}
    if len(mediapipe_hands.multi_hand_landmarks) == 1:
        landmarks = mediapipe_hands.multi_hand_landmarks[0]
        distance_left = calcDistance2D(mediapipe_pose.pose_landmarks.landmark[15], landmarks.landmark[0])
        distance_right = calcDistance2D(mediapipe_pose.pose_landmarks.landmark[16], landmarks.landmark[0])
        left = landmarks if distance_left < distance_right else None
        right = landmarks if distance_left >= distance_right else None
        return {"left": left, "right": right}

    for landmarks in mediapipe_hands.multi_hand_landmarks:
        distance = calcDistance2D(mediapipe_pose.pose_landmarks.landmark[15], landmarks.landmark[0])
        if distance < distance_left:
            distance_left = distance
            left = landmarks
        distance = calcDistance2D(mediapipe_pose.pose_landmarks.landmark[16], landmarks.landmark[0])
        if distance < distance_right:
            distance_right = distance
            right = landmarks

    return {"left": left, "right": right}

def depth_optimizer(mediapipe_pose, mediapipe_hands, scale: Vector2D):
    pose_depth = None
    hand_depth = None
    if mediapipe_pose.pose_landmarks:
        """
        pose_depth = max(
            calcDistance2D(mediapipe_pose.pose_landmarks.landmark[0], mediapipe_pose.pose_landmarks.landmark[1]),
            calcDistance2D(mediapipe_pose.pose_landmarks.landmark[0], mediapipe_pose.pose_landmarks.landmark[4]),
            calcDistance2D(mediapipe_pose.pose_landmarks.landmark[1], mediapipe_pose.pose_landmarks.landmark[4])
        )
        """
        tmp_vector_0 = Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[2])
        tmp_vector_1 = Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[5])
        vector_width = calcDistance2D(tmp_vector_0, tmp_vector_1, scale=scale)
        tmp_vector_0 = calcMiddleVector(Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[2]), Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[5]))
        #tmp_vector_1 = Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[0])
        tmp_vector_1 = calcMiddleVector(Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[9]), Vector2D.fromVector(mediapipe_pose.pose_landmarks.landmark[10]))
        vector_height = calcDistance2D(tmp_vector_0, tmp_vector_1, scale=scale)
        #print(vector_width, vector_height)

        pose_depth = max(vector_width, vector_height)
        _pose_depth = pose_depth / scale.x
        #print(pose_depth)
        for i in range(33):
            mediapipe_pose.pose_landmarks.landmark[i].z = _pose_depth

    if mediapipe_hands.multi_hand_landmarks:
        for i, _ in enumerate(mediapipe_hands.multi_hand_landmarks):
            tmp_vector_0 = Vector2D.fromVector(mediapipe_hands.multi_hand_landmarks[i].landmark[0])
            tmp_vector_5 = Vector2D.fromVector(mediapipe_hands.multi_hand_landmarks[i].landmark[5])
            tmp_vector_17 = Vector2D.fromVector(mediapipe_hands.multi_hand_landmarks[i].landmark[17])
            length_0 = calcDistance2D(tmp_vector_0, tmp_vector_5, scale=scale) * 0.7
            length_1 = calcDistance2D(tmp_vector_0, tmp_vector_17, scale=scale) * 0.8
            length_2 = calcDistance2D(tmp_vector_5, tmp_vector_17, scale=scale)
            #print(max(length_0, length_1, length_2))
            
            hand_depth = max(length_0, length_1, length_2) * 1.1
            _pose_depth = pose_depth if pose_depth else hand_depth
            hand_depth = max(hand_depth, _pose_depth)
            hand_depth = hand_depth + (hand_depth-_pose_depth) * 2.6
            hand_depth = hand_depth / scale.x
            for index in range(21):
                mediapipe_hands.multi_hand_landmarks[i].landmark[index].z = hand_depth
    #if pose_depth and hand_depth:
    #    print(pose_depth/scale.x, hand_depth/scale.x)
    return mediapipe_pose, mediapipe_hands

class Landmarks:
    def __init__(self, *, scale=Vector2D(x=1, y=1)):
        self.pose = self.__pose = None
        self.hands =  self.__hands = None
        self.scale = scale

    def __initialize(self, mediapipe_pose, mediapipe_hands, scale:Vector2D=None):
        if scale:
            self.scale = scale
        if mediapipe_pose.pose_landmarks:
            if not self.__pose:
                self.pose = self.__pose = Pose(mediapipe_pose, scale=self.scale)
            hands = calcRelativeHandsPosition(mediapipe_pose, mediapipe_hands)
            if not self.__hands:
                self.hands = self.__hands = Hands(left=hands["left"], right=hands["right"], scale=self.scale)

    def update(self, mediapipe_pose, mediapipe_hands, scale:Vector2D=None):
        if scale:
            self.scale = scale
        if not self.__pose or not self.__hands:
            self.__initialize(mediapipe_pose, mediapipe_hands)
            return
        depth_optimizer(mediapipe_pose, mediapipe_hands, self.scale)
        self.pose = self.__pose.update(mediapipe_pose, scale=self.scale)
        hands = calcRelativeHandsPosition(mediapipe_pose, mediapipe_hands)
        self.hands = self.__hands.update(left=hands["left"], right=hands["right"], scale=self.scale)

class LandmarkPoint:
    def __init__(self, landmark, *, scale:Vector2D=Vector2D(x=1, y=1)) -> None:
        self.scale = scale
        self.landmark = landmark
        self.raw_landmark = Vector3D(x=self.landmark.x*self.scale.x, y=self.landmark.y*self.scale.y, z=self.landmark.z*self.scale.x)


class ScreenLandmarks:
    def __init__(self, *, eye=Vector3D(x=0, y=0, z=0), origin_point=Vector3D(x=0, y=0, z=0), diagonal_point=Vector3D(x=0, y=0, z=0), horizontal_direction=Vector3D(x=0, y=0, z=0), scale:Vector2D=Vector2D(x=1, y=1)):
        self.eye = LandmarkPoint(eye, scale=scale)
        self.origin_point = LandmarkPoint(origin_point, scale=scale)
        self.diagonal_point = LandmarkPoint(diagonal_point, scale=scale)
        self.horizontal_direction = LandmarkPoint(horizontal_direction)
        self.scale = scale

    def get(self, name: str):
        if name == 'eye':
            return self.eye
        if name == 'origin_point':
            return self.origin_point
        if name == 'diagonal_point':
            return self.diagonal_point
        if name == 'horizontal_direction':
            return self.horizontal_direction
        if name == 'scale':
            return self.scale
        return None

    def set(self, name: str, value):
        if name == 'eye':
            self.eye = value
        elif name == 'origin_point':
            self.origin_point = value
        elif name == 'diagonal_point':
            self.diagonal_point = value
        elif name == 'horizontal_direction':
            self.horizontal_direction = value
        elif name == 'scale':
            self.scale = value