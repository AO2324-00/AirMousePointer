import sys
sys.path.append('../')

import cv2
import mediapipe as mp
import numpy as np

from Commons.Draw import Draw
from Commons.vector import Vector3D, Vector2D, calcMiddleVector, calcVector3D, calcDistance2D
from Commons.landmarks import Landmark, Landmarks, ScreenLandmarks, LandmarkPoint
from UserRecognition.UserRecognition import userRecognition
from VirtualScreen.Calibration import VirtulScreenRecognizer, VirtulScreenEstimator, FixedParameter, linear_function, quadratic_function
from VirtualScreen.VirtulScreen import calcScreenVertex, VirtualScreen, calcPointerPosition
from HandGesture.HandGesture import HandGesture, handGestureRecognition
from Config import ControllerState


class RelativeController:
    def __init__(self, *, controllerState: ControllerState) -> None:
        self.controllerState = controllerState

        self.landmarks = Landmarks()
        self.calibrationRecognizer = VirtulScreenRecognizer()
        self.calibrationEstimator = VirtulScreenEstimator()
        self.relativeVirtualScreen = RelativeVirtualScreen(controllerState.screenLandmarks)
        self.screenLandmarks = None
        self.scale=None
        self.tracking = False
        self.pointer = Landmark(landmark=Vector3D(x=0, y=0, z=0))

    def update(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        Pose, Hands, user_box, self.tracking = userRecognition(image, screen=None, tracking=self.tracking)

        if not self.tracking:
            image = cv2.flip(image, 1)
            return image, None, None, None

        height, width, _ = image.shape
        self.scale = Vector2D(x=width, y=height)
        self.landmarks.update(Pose, Hands, scale=self.scale)

        """
        if not self.landmarks.hands or (not self.landmarks.hands.left and not self.landmarks.hands.right):
            image = cv2.flip(image, 1)
            return image, None, None, None
        """
        
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
            image = Draw.screenPanel(image, user_box)
        if self.landmarks.pose:
            image = Draw.pose(image, self.landmarks.pose)
        if self.landmarks.hands and not self.screenLandmarks:
            image = Draw.hands(image, self.landmarks.hands, thickness=2)

        image, pointer = self.__pointer(image, use_hand)
        if pointer:
            self.pointer.update(landmark=Vector3D(x=pointer.x, y=pointer.y, z=0))
            pointer = self.pointer.get()
            pointer = Vector2D(x=pointer.x, y=pointer.y)

        image = cv2.flip(image, 1)
        return image, pointer, handGesture, hand_dir

    def __calibration(self, image):
        #state, _screenLandmarks = None, None

        state, _screenLandmarks = self.calibrationRecognizer.recognize(self.landmarks)

        if state == 3:
            self.relativeVirtualScreen.calibration(_screenLandmarks)
        if state == 3 or not _screenLandmarks:
            screenLandmarks = self.relativeVirtualScreen.update(calcMiddleVector(self.landmarks.pose.landmark[2].get(), self.landmarks.pose.landmark[5].get()))
            if screenLandmarks:
                self.controllerState.setScreenLandmarks(screenLandmarks)
                self.screenLandmarks = screenLandmarks

        if _screenLandmarks:
            image = Draw.screenPanel(image, calcScreenVertex(_screenLandmarks), color=(150, 245, 250), thickness=2)
        elif self.screenLandmarks:
            image = Draw.screenPanel(image, calcScreenVertex(self.screenLandmarks))

        return image

    def __pointer(self, image, hand):
        if not self.screenLandmarks or not hand:
            return image, None

        vertex = calcScreenVertex(self.screenLandmarks)
        for i, landmarkPoint in enumerate(vertex):
            vertex[i] = LandmarkPoint(Vector3D(x=landmarkPoint.landmark.x, y=landmarkPoint.landmark.y, z=0))
        pointer = LandmarkPoint(Vector3D.fromVector(hand.landmark[5]).set("z", 0), scale=self.landmarks.scale)
        image = Draw.hands(image, self.landmarks.hands, thickness=1)
        image = Draw.point(image, pointer)
        #return image, virtualScreen.calcPointerPosition(pointer)
        return image, calcPointerPosition(vertex, pointer)

class RelativeVirtualScreen:
    def __init__(self, screenLandmarks: ScreenLandmarks) -> None:
        self.__screenLandmarks : ScreenLandmarks = screenLandmarks

    def clear(self):
        self.__screenLandmarks = None
    
    def calibration(self, screenLandmarks: ScreenLandmarks):
        self.__screenLandmarks = screenLandmarks

    def update(self, eye):
        if not self.__screenLandmarks:
            return None
        vector = Vector3D.fromVector(self.__screenLandmarks.eye.landmark).subtraction(eye)
        vector_distance = eye.z - Vector3D.fromVector(self.__screenLandmarks.eye.landmark).z
        center_point = calcMiddleVector(self.__screenLandmarks.origin_point.landmark, self.__screenLandmarks.diagonal_point.landmark)
        origin_vector = calcVector3D(center_point, self.__screenLandmarks.origin_point.landmark)
        origin_distance = calcDistance2D(center_point, self.__screenLandmarks.origin_point.landmark)
        origin_vector = origin_vector.multiply((origin_distance+vector_distance*2)/origin_distance)
        #print(vector_distance, origin_distance, (origin_distance+vector_distance)/origin_distance,  origin_distance*(origin_distance+vector_distance)/origin_distance)
        center_point = center_point.subtraction(vector)
        return ScreenLandmarks(
            eye=eye,
            origin_point=center_point.addition(origin_vector),
            diagonal_point=center_point.subtraction(origin_vector),
            horizontal_direction=Vector3D.fromVector(self.__screenLandmarks.horizontal_direction.landmark),
            scale=self.__screenLandmarks.scale
        )