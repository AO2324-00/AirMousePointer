import sys
sys.path.append('../')

import cv2
import mediapipe as mp
import numpy as np

from Commons.Draw import Draw
from Commons.vector import Vector3D, Vector2D, calcMiddleVector, calcVector3D
from Commons.landmarks import Landmark, Landmarks, ScreenLandmarks, LandmarkPoint
from UserRecognition.UserRecognition import userRecognition
from VirtualScreen.Calibration import VirtulScreenRecognizer, VirtulScreenEstimator, FixedParameter, linear_function, quadratic_function
from VirtualScreen.VirtulScreen import calcScreenVertex, VirtualScreen, calcPointerPosition
from HandGesture.HandGesture import HandGesture, handGestureRecognition
from Config import ControllerState

class Controller:
    def __init__(self, *, controllerState: ControllerState) -> None:
        self.controllerState = controllerState
        self.fixedParameter = controllerState.fixedParameter

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
            return image, None, None, None, None

        height, width, _ = image.shape
        self.scale = Vector2D(x=width, y=height)
        self.landmarks.update(Pose, Hands, scale=self.scale)

        if not self.landmarks.hands or (not self.landmarks.hands.left and not self.landmarks.hands.right):
            image = cv2.flip(image, 1)
            return image, None, None, None, None

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
        return image, pointer, use_hand.landmark[5], handGesture, hand_dir


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
            image = Draw.screenPanel(image, calcScreenVertex(_screenLandmarks), color=(150, 245, 250), thickness=2)
        elif self.screenLandmarks:
            image = image = Draw.screenBox(image, self.screenLandmarks.eye, calcScreenVertex(self.screenLandmarks))
        return image

    def __pointer(self, image, hand):
        if not self.screenLandmarks or not hand:
            return image, None

        virtualScreen = VirtualScreen(self.screenLandmarks)
        pointer = virtualScreen.calcIntersection(self.screenLandmarks.eye.landmark, calcVector3D(self.screenLandmarks.eye.landmark, hand.landmark[5]))
        image = Draw.hands(image, self.landmarks.hands, thickness=1)
        image = Draw.point(image, pointer)
        image = Draw.line(image, self.screenLandmarks.eye, LandmarkPoint(hand.landmark[5], scale=self.landmarks.scale))
        #return image, virtualScreen.calcPointerPosition(pointer)
        return image, calcPointerPosition(virtualScreen.getVertex(), pointer)
        
    def startCalibration(self):
        self.fixedParameter = None
        self.screenLandmarks = None
    
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