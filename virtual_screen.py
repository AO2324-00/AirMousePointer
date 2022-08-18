import time as Timer
from typing import Optional
import numpy as np
from landmarks import Landmarks, BothSides
import vector
from vector import Vector3D
from screen import SpatialPlane, calcVertex


class VirtualScreen:

    def __init__(self, *, calibration_frame_limit: int=5):
        self.__isCalibrating: bool = False
        self.__spatial_plane: Optional[SpatialPlane] = None

        self.__calibration_frame_limit: int = calibration_frame_limit

        self.__time: Optional[float] = None
        self.__screen_positions = np.array([])
        
    
    def update(self, landmark: Landmarks) -> None:
        self.__isCalibrating = self.__calibration(landmark)
        if self.__isCalibrating:
            vertex = self.__calcVertex(landmark)
            if vertex:
                self.__spatial_plane = SpatialPlane(vertex)
        else:
            self.__time = None
            self.__screen_positions = np.array([])
    
    def isCalibrating(self) -> bool:
        return self.__isCalibrating

    def getSpatialPlane(self) -> Optional[SpatialPlane]:
        return self.__spatial_plane

    def __calibration(self, landmarks: Landmarks) -> bool:

        if not (landmarks.eye and landmarks.hands.left and landmarks.hands.right) :
            return False

        averaged_landmarks = landmarks.averaged(self.__calibration_frame_limit)

        hand_vector = BothSides(
            left=vector.calcVector3D(averaged_landmarks.hands.get('left').landmark[2], averaged_landmarks.hands.get('left').landmark[4]),
            right=vector.calcVector3D(averaged_landmarks.hands.get('right').landmark[4], averaged_landmarks.hands.get('right').landmark[2]))
        hand_index_vector = BothSides(
            left=vector.calcVector3D(averaged_landmarks.hands.get('left').landmark[2], averaged_landmarks.hands.get('left').landmark[4]),
            right=vector.calcVector3D(averaged_landmarks.hands.get('right').landmark[2], averaged_landmarks.hands.get('right').landmark[4]))
        hand_thumb_vector = BothSides(
            left=vector.calcVector3D(averaged_landmarks.hands.get('left').landmark[5], averaged_landmarks.hands.get('left').landmark[8]),
            right=vector.calcVector3D(averaged_landmarks.hands.get('right').landmark[5], averaged_landmarks.hands.get('right').landmark[8]))

        index_angle = vector.calcDotProduct(hand_vector.left, hand_vector.right)
        hand_angle = BothSides(
            left=vector.calcDotProduct(hand_index_vector.left, hand_thumb_vector.left),
            right=vector.calcDotProduct(hand_index_vector.right, hand_thumb_vector.right))
        is_orthogonal = BothSides(left= 60 < hand_angle.left < 110, right= 60 < hand_angle.right < 110)

        return index_angle < 45 and is_orthogonal.left and is_orthogonal.right


    def __calcVertex(self, landmarks: Landmarks) -> Optional[np.ndarray]:

        if not self.__time:
            self.__time = Timer.time()

        elapsed_time = Timer.time() - self.__time
        
        if elapsed_time < 1:
            return None
        if 2 < elapsed_time:
            getVector = np.vectorize(lambda frame, id: frame[id])
            screen_vertex = calcVertex(
                vector.Average3D(getVector(self.__screen_positions, 'v')),
                vector.Average3D(getVector(self.__screen_positions, 'left')),
                vector.Average3D(getVector(self.__screen_positions, 'right')))
            self.__screen_positions = np.array([])
            self.__time = None
            return screen_vertex

        v = vector.calcVector3D(
            vector.calcVector3D(landmarks.hands.right.landmark[5], landmarks.hands.right.landmark[8]),
            vector.calcVector3D(landmarks.hands.left.landmark[5], landmarks.hands.left.landmark[8])
        )
        position = {'v': v, 'left': landmarks.hands.left.landmark[1], 'right': landmarks.hands.right.landmark[1]}
        
        self.__screen_positions = np.append(self.__screen_positions, position)

        screen_vertex = calcVertex(v, position['left'], position['right'])
        return None
