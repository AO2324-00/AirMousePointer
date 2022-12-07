import sys
sys.path.append('../')

import cv2
import mediapipe as mp
import numpy as np

from Commons.vector import Vector3D, Vector2D, calcMiddleVector, calcVector3D
from Commons.landmarks import Landmark, Landmarks, ScreenLandmarks, LandmarkPoint
from UserRecognition.UserRecognition import userRecognition
from VirtualScreen.Calibration import VirtulScreenRecognizer, VirtulScreenEstimator, FixedParameter, linear_function, quadratic_function
from VirtualScreen.VirtulScreen import calcScreenVertex, VirtualScreen
from HandGesture.HandGesture import HandGesture, handGestureRecognition

hands =  mp.solutions.hands.Hands(
    static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    max_num_hands = 4,              # 認識する手の最大数。
    model_complexity = 1,           # 手のランドマークモデルの複雑さ(0 or 1)。
    #min_detection_confidence = 0.3,
    min_detection_confidence = 0.5, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
    min_tracking_confidence = 0.5   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
)
pose = mp.solutions.pose.Pose(
    static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    model_complexity = 2,           # ランドマークモデルの複雑さ(0 or 1 or 2)。
    min_detection_confidence = 0.5, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
    min_tracking_confidence = 0.5   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
)

def draw_border(image, screen_vertex, color=(255, 255, 255), thickness=2):
    annotated_image = image.copy()
    for index in range(len(screen_vertex)):
        #print((int(screen_vertex[index-1].raw_landmark.x), int(screen_vertex[index-1].raw_landmark.y)))
        cv2.line(annotated_image, (int(screen_vertex[index-1].raw_landmark.x), int(screen_vertex[index-1].raw_landmark.y)), (int(screen_vertex[index].raw_landmark.x), int(screen_vertex[index].raw_landmark.y)), color, thickness)
    return annotated_image

def draw_pose_landmarks(image, landmarks, color=(100, 110, 104), thickness=2):
    annotated_image = image.copy()
    if not landmarks:
        return annotated_image
    for index in mp.solutions.pose.POSE_CONNECTIONS:
        cv2.line(annotated_image, (int(landmarks.raw_landmark[index[0]].x), int(landmarks.raw_landmark[index[0]].y)), (int(landmarks.raw_landmark[index[1]].x), int(landmarks.raw_landmark[index[1]].y)), color, thickness)
    return annotated_image

def draw_hands_landmarks(image, landmarks, color=(15, 50, 255), thickness=2):
    annotated_image = image.copy()
    for dir in ["left", "right"]:
        if not landmarks or not landmarks.get(dir):
            continue
        for index in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(annotated_image, (int(landmarks.get(dir).raw_landmark[index[0]].x), int(landmarks.get(dir).raw_landmark[index[0]].y)), (int(landmarks.get(dir).raw_landmark[index[1]].x), int(landmarks.get(dir).raw_landmark[index[1]].y)), color, thickness)
    return annotated_image

def draw_point(image, point_landmark, color=(150, 245, 250)):
    annotated_image = image.copy()
    cv2.drawMarker(annotated_image, (int(point_landmark.raw_landmark.x), int(point_landmark.raw_landmark.y)), color=color, markerType=cv2.MARKER_TRIANGLE_UP, thickness=2, markerSize=10)
    return annotated_image

def draw_line(image, line_landmarks, color=(255, 255, 255), thickness=1):
    annotated_image = image.copy()
    cv2.line(annotated_image, (int(line_landmarks[0].raw_landmark.x), int(line_landmarks[0].raw_landmark.y)), (int(line_landmarks[1].raw_landmark.x), int(line_landmarks[1].raw_landmark.y)), color, thickness)
    return annotated_image

class ControllerState:
    def __init__(self, lefty=False) -> None:
        self.calibrating = False
        self.lefty = lefty

class Controller:
    def __init__(self, *, controllerState: ControllerState, fixedParameter: FixedParameter=None) -> None:
        self.controllerState = controllerState
        self.fixedParameter = fixedParameter

        self.landmarks = Landmarks()
        self.calibrationRecognizer = VirtulScreenRecognizer()
        self.calibrationEstimator = VirtulScreenEstimator()
        self.screenLandmarks = None
        self.scale=None
        self.tracking = False
        self.pointer = Landmark(landmark=Vector3D(x=0, y=0, z=0))

    def update(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        Pose, Hands, user_box, self.tracking = userRecognition(image, screen=self.fixedParameter, tracking=self.tracking)

        if not self.tracking:
            image = cv2.flip(image, 1)
            return image, None, None, None

        height, width, _ = image.shape
        self.scale = Vector2D(x=width, y=height)
        self.landmarks.update(Pose, Hands, scale=self.scale)

        if not self.landmarks.hands or (not self.landmarks.hands.left and not self.landmarks.hands.right):
            image = cv2.flip(image, 1)
            return image, None, None, None

        image = self.__calibration(image)

        use_hand = None
        hand_dir = None
        if self.landmarks.hands:
            main_hand = self.landmarks.hands.left if self.controllerState.lefty else self.landmarks.hands.right
            sub_hand = self.landmarks.hands.right if self.controllerState.lefty else self.landmarks.hands.left
            use_hand = main_hand if main_hand else sub_hand
            if main_hand:
                hand_dir = "left" if self.controllerState.lefty else "right"
            elif sub_hand:
                hand_dir = "right" if self.controllerState.lefty else "left"

        

        handGesture: HandGesture = handGestureRecognition(use_hand)

        if user_box is not None:
            image = draw_border(image, user_box)
        if self.landmarks.pose:
            image = draw_pose_landmarks(image, self.landmarks.pose)
        if self.landmarks.hands:
            image = draw_hands_landmarks(image, self.landmarks.hands)

        image, pointer = self.__pointer(image, use_hand)
        if pointer:
            self.pointer.update(landmark=Vector3D(x=pointer.x, y=pointer.y, z=0))
            pointer = self.pointer.get()
            pointer = Vector2D(x=pointer.x, y=pointer.y)

        image = cv2.flip(image, 1)
        return image, pointer, handGesture, hand_dir


    def __calibration(self, image):
        state, _screenLandmarks = None, None
        if self.controllerState.calibrating:
            state, _screenLandmarks = self.calibrationRecognizer.recognize(self.landmarks)

        if state == 3:
            self.fixedParameter = self.calibrationEstimator.estimation(_screenLandmarks)
        if state == 3 or not _screenLandmarks:
            screenLandmarks = self.calibrationEstimator.calcScreenLandmarks(self.landmarks, self.fixedParameter)
            self.screenLandmarks = screenLandmarks if screenLandmarks else self.screenLandmarks
        if _screenLandmarks:
            image = draw_border(image, calcScreenVertex(_screenLandmarks), color=(150, 245, 250))
        elif self.screenLandmarks:
            image = draw_border(image, calcScreenVertex(self.screenLandmarks))
        return image

    def __pointer(self, image, hand):
        if not self.screenLandmarks or not hand:
            return image, None

        virtualScreen = VirtualScreen(self.screenLandmarks)
        pointer = virtualScreen.calcIntersection(self.screenLandmarks.eye.landmark, calcVector3D(self.screenLandmarks.eye.landmark, hand.landmark[5]))
        image = draw_point(image, pointer)
        image = draw_line(image, (self.screenLandmarks.eye, LandmarkPoint(hand.landmark[5], scale=self.landmarks.scale)))
        return image, virtualScreen.calcPointerPosition(pointer)
        
    
    def applyCalibration(self):
        if not self.controllerState.calibrating:
            return None
        self.controllerState.calibrating = False
        return self.fixedParameter
    
    def cancelCalibration(self, fixedParameter: FixedParameter):
        if not self.controllerState.calibrating:
            return False
        self.fixedParameter = fixedParameter
        self.controllerState.calibrating = False
        return True