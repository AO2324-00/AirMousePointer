import numpy as np
from scipy.optimize import curve_fit
import time as Timer
from typing import Optional

from Commons.vector import Vector2D, Vector3D, calcVector2D, calcVector3D, calcDotProduct, calcMiddleVector, calcCornerVector
from Commons.landmarks import Landmarks, ScreenLandmarks

class VirtulScreenRecognizer:

    def __init__(self):
        self.__time: Optional[float] = Timer.time()
        self.__ScreenLandmarks_list = np.array([])

    def recognize(self, landmarks: Landmarks):
        isRecognizing = self.__isRecognizing(landmarks)
        if not isRecognizing:
            self.__time = Timer.time()
            self.__ScreenLandmarks_list = np.array([])
            return 0, None
        elapsed_time = Timer.time() - self.__time
        if elapsed_time < 1:
            return 1, None
        if 3 < elapsed_time:
            screenLandmarks = self.__calcAverage()
            self.__time = Timer.time()
            self.__ScreenLandmarks_list = np.array([])
            return 3, screenLandmarks
        screenLandmarks = self.__calcScreenLandmarks(landmarks)
        self.__ScreenLandmarks_list = np.append(self.__ScreenLandmarks_list, screenLandmarks)
        return 2, screenLandmarks

    def __isRecognizing(self, landmarks: Landmarks) -> bool:
        if not landmarks.hands or not landmarks.hands.left or not landmarks.hands.right or not landmarks.pose:
            return False
        index_left = calcVector2D(landmarks.hands.left.raw_landmark[5], landmarks.hands.left.raw_landmark[8])
        index_right = calcVector2D(landmarks.hands.right.raw_landmark[5], landmarks.hands.right.raw_landmark[8])
        thumb_left = calcVector2D(landmarks.hands.left.raw_landmark[2], landmarks.hands.left.raw_landmark[4])
        thumb_right = calcVector2D(landmarks.hands.right.raw_landmark[2], landmarks.hands.right.raw_landmark[4])
        perpendicular_left = 50 < calcDotProduct(index_left, thumb_left) < 110
        perpendicular_right = 50 < calcDotProduct(index_right, thumb_right) < 110
        parallel = 150 < calcDotProduct(index_left, index_right)
        return perpendicular_left and perpendicular_right and parallel

    def __calcScreenLandmarks(self, landmarks: Landmarks)->ScreenLandmarks:

        eye = calcMiddleVector(landmarks.pose.landmark[2], landmarks.pose.landmark[5])
        raw_point_left = landmarks.hands.left.raw_landmark[5]
        raw_point_right = landmarks.hands.right.raw_landmark[5]

        vector_left = calcVector3D(landmarks.hands.left.landmark[5].get(), landmarks.hands.left.landmark[8].get())
        vector_right = calcVector3D(landmarks.hands.right.landmark[8].get(), landmarks.hands.right.landmark[5].get())
        vector = vector_left.addition(vector_right)
        vector.z = calcVector3D(landmarks.hands.left.landmark[5].get(), landmarks.hands.right.landmark[5].get()).z

        if raw_point_left.y < raw_point_right.y:
            return ScreenLandmarks(
                eye=eye,
                origin_point=landmarks.hands.left.landmark[5].get(),
                diagonal_point=landmarks.hands.right.landmark[5].get(),
                horizontal_direction=vector,
                scale=landmarks.scale
            )

        raw_vector_left = calcVector3D(landmarks.hands.left.raw_landmark[5], landmarks.hands.left.raw_landmark[8])
        raw_vector_right = calcVector3D(landmarks.hands.right.raw_landmark[8], landmarks.hands.right.raw_landmark[5])
        raw_vector = raw_vector_left.addition(raw_vector_right)
        raw_vector.z = calcVector3D(raw_point_left, raw_point_right).z

        raw_origin_point = calcCornerVector(raw_point_left, raw_point_right, raw_vector)
        raw_diagonal_point = calcCornerVector(raw_point_right, raw_point_left, raw_vector)
        origin_point = Vector3D(x=raw_origin_point.x/landmarks.scale.x, y=raw_origin_point.y/landmarks.scale.y, z=raw_origin_point.z/landmarks.scale.x)
        diagonal_point = Vector3D(x=raw_diagonal_point.x/landmarks.scale.x, y=raw_diagonal_point.y/landmarks.scale.y, z=raw_diagonal_point.z/landmarks.scale.x)
        
        return ScreenLandmarks(
            eye=eye,
            origin_point=origin_point,
            diagonal_point=diagonal_point,
            horizontal_direction=vector,
            scale=landmarks.scale
        )

    def __calcAverage(self):
        result = {
            "eye": Vector3D(x=0,y=0,z=0),
            "origin_point": Vector3D(x=0,y=0,z=0),
            "diagonal_point": Vector3D(x=0,y=0,z=0),
            "horizontal_direction": Vector3D(x=0,y=0,z=0)
        }
        for screen_landmarks in self.__ScreenLandmarks_list:
            result["eye"] = result["eye"].addition(screen_landmarks.eye.landmark)
            result["origin_point"] = result["origin_point"].addition(screen_landmarks.origin_point.landmark)
            result["diagonal_point"] = result["diagonal_point"].addition(screen_landmarks.diagonal_point.landmark)
            result["horizontal_direction"] = result["horizontal_direction"].addition(screen_landmarks.horizontal_direction.landmark)
        size = len(self.__ScreenLandmarks_list)
        result["eye"] = result["eye"].division(size)
        result["origin_point"] = result["origin_point"].division(size)
        result["diagonal_point"] = result["diagonal_point"].division(size)
        result["horizontal_direction"] = result["horizontal_direction"].division(size)
        return ScreenLandmarks(
            eye=result["eye"],
            origin_point=result["origin_point"],
            diagonal_point=result["diagonal_point"],
            horizontal_direction=result["horizontal_direction"],
            scale=self.__ScreenLandmarks_list[0].scale
        )

def linear_function(X, a, b):
    return a*X[0] + b

def quadratic_function(X, a, b):
    return a*X[0]*X[0] + b

class Parameter3D:

    def __init__(self, *, x=None, y=None, z=None) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.y}"

class FixedParameter:
    def __init__(self, *, centor_point: Parameter3D, origin_vector: Parameter3D, horizontal_direction: Parameter3D) -> None:
        self.centor_point = centor_point
        self.origin_vector = origin_vector
        self.horizontal_direction = horizontal_direction

class VirtulScreenEstimator:

    def __init__(self, history_limit:int=100):
        self.history_limit = history_limit
        self.__ScreenLandmarks_list = np.array([])
        pass
    
    def estimation(self, screenLandmarks: ScreenLandmarks):
        if not screenLandmarks:
            return None
        if self.history_limit < len(self.__ScreenLandmarks_list):
            self.__ScreenLandmarks_list = np.delete(self.__ScreenLandmarks_list, 0)
        self.__ScreenLandmarks_list = np.append(self.__ScreenLandmarks_list, screenLandmarks)
        if len(self.__ScreenLandmarks_list) < 3:
            return None
        return self.__calcParameter()

    def calcScreenLandmarks(self, landmarks: Landmarks, fixedParameter: FixedParameter):
        if not fixedParameter or not landmarks or not landmarks.pose:
            return None
        eye = calcMiddleVector(landmarks.pose.landmark[2], landmarks.pose.landmark[5])
        center_point = Vector3D(
            x=linear_function(np.array([eye.x]), fixedParameter.centor_point.x[0], fixedParameter.centor_point.x[1]),
            y=linear_function(np.array([eye.y]), fixedParameter.centor_point.y[0], fixedParameter.centor_point.y[1]),
            z=linear_function(np.array([eye.z]), fixedParameter.centor_point.z[0], fixedParameter.centor_point.z[1])
        )
        origin_vector = Vector3D(
            x=quadratic_function(np.array([eye.z]), fixedParameter.origin_vector.x[0], fixedParameter.origin_vector.x[1]),
            y=quadratic_function(np.array([eye.z]), fixedParameter.origin_vector.y[0], fixedParameter.origin_vector.y[1]),
            z=quadratic_function(np.array([eye.z]), fixedParameter.origin_vector.z[0], fixedParameter.origin_vector.z[1])
        )
        return ScreenLandmarks(
            eye=eye,
            origin_point=center_point.addition(origin_vector),
            diagonal_point=center_point.subtraction(origin_vector),
            horizontal_direction=fixedParameter.horizontal_direction,
            scale=landmarks.scale
        )

    def __calcParameter(self):
        eye = Parameter3D(x=np.array([]), y=np.array([]), z=np.array([]))
        centor_point = Parameter3D(x=np.array([]), y=np.array([]), z=np.array([]))
        origin_vector = Parameter3D(x=np.array([]), y=np.array([]), z=np.array([]))
        horizontal_direction = Parameter3D(x=np.array([]), y=np.array([]), z=np.array([]))
        for screenLandmarks in self.__ScreenLandmarks_list:
            eye.x = np.append(eye.x, screenLandmarks.eye.landmark.x)
            eye.y = np.append(eye.y, screenLandmarks.eye.landmark.y)
            eye.z = np.append(eye.z, screenLandmarks.eye.landmark.z)
            _centor_point = calcMiddleVector(screenLandmarks.origin_point.landmark, screenLandmarks.diagonal_point.landmark)
            centor_point.x = np.append(centor_point.x, _centor_point.x)
            centor_point.y = np.append(centor_point.y, _centor_point.y)
            centor_point.z = np.append(centor_point.z, _centor_point.z)
            origin_vector.x = np.append(origin_vector.x, calcVector3D(_centor_point, screenLandmarks.origin_point.landmark).x)
            origin_vector.y = np.append(origin_vector.y, calcVector3D(_centor_point, screenLandmarks.origin_point.landmark).y)
            origin_vector.z = np.append(origin_vector.z, calcVector3D(_centor_point, screenLandmarks.origin_point.landmark).z)
            horizontal_direction.x = np.append(horizontal_direction.x, screenLandmarks.horizontal_direction.landmark.x)
            horizontal_direction.y = np.append(horizontal_direction.y, screenLandmarks.horizontal_direction.landmark.y)
            horizontal_direction.z = np.append(horizontal_direction.z, screenLandmarks.horizontal_direction.landmark.z)

        return FixedParameter(
            centor_point=Parameter3D(
                x=curve_fit(linear_function, np.array([eye.x]), centor_point.x)[0],
                y=curve_fit(linear_function, np.array([eye.y]), centor_point.y)[0],
                z=curve_fit(linear_function, np.array([eye.z]), centor_point.z)[0]
            ),
            origin_vector=Parameter3D(
                x=curve_fit(quadratic_function, np.array([eye.z]), origin_vector.x)[0],
                y=curve_fit(quadratic_function, np.array([eye.z]), origin_vector.y)[0],
                z=curve_fit(quadratic_function, np.array([eye.z]), origin_vector.z)[0]
            ),
            horizontal_direction=Vector3D(
                x=np.average(horizontal_direction.x),
                y=np.average(horizontal_direction.y),
                z=np.average(horizontal_direction.z)
            )
        )
