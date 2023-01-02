import mouse
import keyboard
import numpy as np

from Commons.vector import Vector2D, calcDistance2D, calcVector2D
from Controller.Controller import HandGesture, ControllerState

def calcPointerPosition(pointer: Vector2D, screen_size: Vector2D):
    return Vector2D(x=pointer.x*screen_size.x, y=pointer.y*screen_size.y)

class MouseEvent:
    def __init__(self, controllerState: ControllerState, screen_size: Vector2D, pointer_offset:Vector2D=Vector2D(x=350, y=60)) -> None:
        self.controllerState = controllerState
        self.screen_size: Vector2D = screen_size
        self.pointer_offset = pointer_offset
        self.scroll_origin = None
        self.relative_origin = None
        self._relative_origin = None
        self.__button0 = False
        self.__button1 = False
    def update(self, pointer: Vector2D, _pointer: Vector2D, handGesture: HandGesture, hand_dir: str):
        if not pointer or not handGesture:
            return
        pointer_max_distance = calcDistance2D(Vector2D(x=0, y=0), self.screen_size)*0.5
        pointer = calcPointerPosition(pointer, self.screen_size)
        _pointer = calcPointerPosition(_pointer, self.screen_size)

        if hand_dir == "right":
            pointer = pointer.subtraction(self.pointer_offset)
        elif hand_dir == "left":
            pointer = pointer.addition(self.pointer_offset)

        
        if self.__scroll(pointer, handGesture):
            return
        if not self.__relative(pointer, _pointer, handGesture):

            if not (-pointer_max_distance < pointer.x < self.screen_size.x + pointer_max_distance and -pointer_max_distance < pointer.y < self.screen_size.y + pointer_max_distance):
                return
            if pointer.x < 0:
                pointer.x = 0
            if pointer.y < 0:
                pointer.y = 0
            if self.screen_size.x < pointer.x+10:
                pointer.x = self.screen_size.x-10
            if self.screen_size.y < pointer.y+10:
                pointer.y = self.screen_size.y-10

            mouse.move(int(pointer.x), int(pointer.y), True)

        if self.controllerState.calibrating:
            return
        
        if handGesture.button0:
            mouse.press(button="left")
            self.__button0 = True
        elif self.__button0:
            mouse.release(button="left")
            self.__button0 = False

        if handGesture.button1:
            mouse.press(button="right")
            self.__button1 = True
        elif self.__button1:
            mouse.release(button="right")
            self.__button1 = False
        

    def __scroll(self, pointer: Vector2D, handGesture: HandGesture):
        if not handGesture.scroll:
            self.scroll_origin = None
            return False
        if not self.scroll_origin:
            self.scroll_origin = pointer
            return True
        if self.controllerState.calibrating:
            self.scroll_origin = None
            return True
        vector = pointer.subtraction(self.scroll_origin)
        if abs(vector.x) < abs(vector.y):
            mouse.wheel(delta=np.sign(vector.y)*-1)
        else:
            keyboard.press('shift')
            mouse.wheel(delta=np.sign(vector.y)*-1)
            keyboard.release('shift')
        return True
    
    def __relative(self, pointer: Vector2D, _pointer: Vector2D, handGesture: HandGesture, amp=0.25):
        if not handGesture.relative:
            self.relative_origin = None
            self._relative_origin = None
            return False
        if not self.relative_origin:
            self.relative_origin = pointer
            self._relative_origin = _pointer
            return True
        delta = calcVector2D(self._relative_origin, _pointer)
        #print(delta)
        pointer = self.relative_origin.addition(delta.multiply(amp))
        #print(pointer)
        mouse.move(int(pointer.x), int(pointer.y), True)
        return True

        
