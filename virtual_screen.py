import time as Timer
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from torch import le
from landmarks import Landmarks, BothSides
import vector
from vector import Vector3D, calcMiddleVector, calcVector3D
from screen import SpatialPlane, calcVertex

def func(X, a, b, c):
    return X[0] * a + X[1] * b + X[2] * c

class VirtualScreen:

    def __init__(self, *, calibration_frame_limit: int=5, calibration_history_limit: int=3):
        self.__isCalibrating: bool = False
        self.__spatial_plane: Optional[SpatialPlane] = None

        self.__calibration_frame_limit: int = calibration_frame_limit

        self.__time: Optional[float] = None
        self.__screen_positions = np.array([])

        self.__calibration_history_limit = calibration_history_limit
        self.__screen_positions_history = np.array([])
        self.__param_fixed = None
        
    
    def update(self, landmarks: Landmarks) -> None:
        self.__isCalibrating = self.__calibration(landmarks)
        if self.__isCalibrating:
            #vertex = self.__calcVertex(landmarks)
            positions = self.__calcVertex(landmarks)
            #print(self.positions)
            if positions:
                self.__param_fixed = self.__calcCalibratedFunction(positions)
                vertex = calcVertex(positions['v'], positions['left'], positions['right'])
                self.__spatial_plane = SpatialPlane(vertex)
            #if vertex:
            #    self.__spatial_plane = SpatialPlane(vertex)
        else:
            self.__time = None
            self.__screen_positions = np.array([])
            if self.__param_fixed and landmarks.eye:
                result = {}
                for side in ['v', 'left', 'right']:
                    result[side] = Vector3D(
                        x=self.__param_fixed[side]['x'][0]*landmarks.eye.x + self.__param_fixed[side]['x'][1]*landmarks.eye.y + self.__param_fixed[side]['x'][2]*landmarks.eye.z,
                        y=self.__param_fixed[side]['y'][0]*landmarks.eye.x + self.__param_fixed[side]['y'][1]*landmarks.eye.y + self.__param_fixed[side]['y'][2]*landmarks.eye.z,
                        z=self.__param_fixed[side]['z'][0]*landmarks.eye.x + self.__param_fixed[side]['z'][1]*landmarks.eye.y + self.__param_fixed[side]['z'][2]*landmarks.eye.z,)
                vertex = calcVertex(result['v'], result['left'], result['right'])
                self.__spatial_plane = SpatialPlane(vertex)
    
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


    #def __calcVertex(self, landmarks: Landmarks) -> Optional[np.ndarray]:
    def __calcVertex(self, landmarks: Landmarks):

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
            positions = {'eye': landmarks.eye, 'v': vector.Average3D(getVector(self.__screen_positions, 'v')), 'left': vector.Average3D(getVector(self.__screen_positions, 'left')), 'right': vector.Average3D(getVector(self.__screen_positions, 'right'))}
            self.__screen_positions = np.array([])
            self.__time = None
            #return screen_vertex
            return positions

        v = vector.calcVector3D(
            vector.calcVector3D(landmarks.hands.right.landmark[5], landmarks.hands.right.landmark[8]),
            vector.calcVector3D(landmarks.hands.left.landmark[5], landmarks.hands.left.landmark[8])
        )
        position = {'v': v, 'left': landmarks.hands.left.landmark[1], 'right': landmarks.hands.right.landmark[1]}
        
        self.__screen_positions = np.append(self.__screen_positions, position)

        #screen_vertex = calcVertex(v, position['left'], position['right'])
        return None

    def __calcCalibratedFunction(self, positions):
        self.__screen_positions_history = np.append(self.__screen_positions_history, positions)
        if len(self.__screen_positions_history) > self.__calibration_history_limit:
            self.__screen_positions_history = np.delete(self.__screen_positions_history, 0)
        print("キャリブレートカウント: ",len(self.__screen_positions_history))

        if len(self.__screen_positions_history) <= 2:
            return None
        input_array = {'eye': np.empty((3, 0)), 'positions': {'left': {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}, 'right': {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}, 'v': {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}}}
        for history in self.__screen_positions_history:
            tmp = history['eye'].parseArray()
            input_array['eye'] = np.append(input_array['eye'], np.array([[tmp[0]],[tmp[1]],[tmp[2]]]), axis=1)
            for side in ['left', 'right', 'v']:
                for dir in ['x', 'y', 'z']:
                    input_array['positions'][side][dir] = np.append(input_array['positions'][side][dir], history[side].get(dir))

        result = {'v': {}, 'left': {}, 'right': {}}
        for side in ['v', 'left', 'right']:
            for dir in ['x', 'y', 'z']:
                param_fixed, _ = curve_fit(func, input_array['eye'], input_array['positions'][side][dir])
                result[side][dir] = param_fixed

        return result
