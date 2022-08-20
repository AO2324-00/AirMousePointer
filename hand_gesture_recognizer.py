
from typing import Optional
from landmarks import BothSides, HandLandmarks, Landmarks
from vector import calcDistance3D, calcDotProduct, calcVector3D


class HandGestureRecognizer:
    def __init__(self):
        self.__state = {'left': False, 'right': False, 'middle': False}
        self.__is_mouse_moving = BothSides(left=False, right=False)
        self.__is_mouse_pressed = BothSides(
            left={'left': False, 'right': False, 'middle': False}, 
            right={'left': False, 'right': False, 'middle': False})
        self.__is_mouse_scrolling = BothSides(left=False, right=False)

    def update(self, landmarks: Landmarks):
        for side in ['left', 'right']:
            if not landmarks.hands.get(side):
                continue
            self.__is_mouse_moving.set(side, self.__isMouseMoving(landmarks.hands.get(side)))
            for type in ['left', 'right', 'middle']:
                self.__is_mouse_pressed.get(side)[type] = self.__isMousePressed(landmarks.hands.get(side), type=type)
            # if side == "right":
            self.__is_mouse_scrolling.set(side, self.__isScrolling(landmarks.hands.get(side)))

    def getState(self, name: str) -> bool:
        return self.__state[name]
    
    def setState(self, name: str, state: bool):
        self.__state[name] = state

    def isMouseMoving(self):
        return self.__is_mouse_moving

    def isMousePressed(self):
        return self.__is_mouse_pressed

    def isMouseScrolling(self):
        return self.__is_mouse_scrolling


    def __isMouseMoving(self, hand: HandLandmarks):
        palm = calcVector3D(hand.landmark[0], hand.landmark[17])
        finger = calcVector3D(hand.landmark[18], hand.landmark[19])
        angle = calcDotProduct(palm, finger)
        return 150 < angle < 190

    def __isMousePressed(self, hand: HandLandmarks, *, type: str):
        threshold = calcDistance3D(hand.landmark[2], hand.landmark[3]) * 0.9
        distance = None
        if type == "left":
            distance = calcDistance3D(hand.landmark[4], hand.landmark[8])
        if type == "right":
            distance = calcDistance3D(hand.landmark[4], hand.landmark[12])
        if type == "middle":
            distance = calcDistance3D(hand.landmark[4], hand.landmark[16])
        return distance < threshold

    def __isScrolling(self, hand: HandLandmarks):
        palm_0 = calcVector3D(hand.landmark[0], hand.landmark[5])
        palm_1 = calcVector3D(hand.landmark[0], hand.landmark[9])
        palm_3 = calcVector3D(hand.landmark[0], hand.landmark[17])
        finger_mid_0 = calcVector3D(hand.landmark[5], hand.landmark[7])
        finger_mid_1 = calcVector3D(hand.landmark[9], hand.landmark[11])
        finger_mid_3 = calcVector3D(hand.landmark[17], hand.landmark[19])
        finger_angle_0 = calcDotProduct(palm_0, finger_mid_0)
        finger_angle_1 = calcDotProduct(palm_1, finger_mid_1)
        finger_angle_3 = calcDotProduct(palm_3, finger_mid_3)
        finger_angle_4 = calcDotProduct(calcVector3D(hand.landmark[0], hand.landmark[1]), calcVector3D(hand.landmark[2], hand.landmark[4]))
        angle_min = 90
        is_closed_finger_0 = finger_angle_0 > angle_min
        is_closed_finger_1 = finger_angle_1 > angle_min
        is_closed_finger_3 = finger_angle_3 > angle_min
        thumb = finger_angle_4 < 38
        
        return is_closed_finger_0 and is_closed_finger_1 and is_closed_finger_3 and thumb

