import time

import cv2
from numpy import sign
import mouse
import keyboard
from screeninfo import get_monitors

from  estimate import Estimate
from vector import Vector2D, Smoothing2D, calcVector2D

def mouseController(config, estimate: Estimate, smoothing: Smoothing2D):
    position = None
    positions = estimate.pointer.getPositions()
    #print(positions)
    side = estimate.pointer.getActiveSide(positions)
    if not side:
        return

    # スクロール
    #print(estimate.hand_gesture_recognizer.isMouseScrolling().get(side))
    if estimate.hand_gesture_recognizer.isMouseScrolling().get(side):
        if not estimate.pointer.isSaving('scroller'):
            #print("scroll start")
            estimate.pointer.savePositions('scroller')
        v = calcVector2D(estimate.pointer.getSavePositions('scroller', True).get(side),estimate.pointer.getPositions(True).get(side))
        #print(v, estimate.pointer.getSavePositions('scroller', True).get(side))
        if estimate.pointer.getSavePositions('scroller', True).get(side):
            if abs(v.x) < abs(v.y):
                mouse.wheel(delta=sign(v.y)*-1)
            else:
                keyboard.press('shift')
                mouse.wheel(delta=sign(v.x)*-1)
                keyboard.release('shift')
        return
    elif estimate.pointer.isSaving('scroller'):
        #print("scroll end")
        estimate.pointer.releaseSavePositions('scroller')
    
    # ポインタの移動
    if estimate.hand_gesture_recognizer.isMouseMoving().get(side):
        if not estimate.pointer.isSaving('pointer'):
            estimate.pointer.savePositions('pointer')
        v = calcVector2D(estimate.pointer.getSavePositions('pointer', True).get(side),estimate.pointer.getPositions(True).get(side))
        if estimate.pointer.getSavePositions('pointer').get(side):
            position = estimate.pointer.getSavePositions('pointer').get(side).addition(v.multiply(config.fine_mouse_sensitivity))
    else:
        if estimate.pointer.isSaving('pointer') and not estimate.hand_gesture_recognizer.isMouseScrolling().get(side):
            estimate.pointer.releaseSavePositions('pointer')
        position = positions.get(side)

    if not position:
        return

    position = Vector2D(x=position.x*config.screen_size.x, y=position.y*config.screen_size.y)
    smoothing.update(position)
    smoothed = smoothing.get()
    mouse.move(smoothed.x, smoothed.y, True)

    # ボタンの入力
    for button in ["left", "right", "middle"]:
        if estimate.hand_gesture_recognizer.isMousePressed().get(side)[button]:
            if not estimate.hand_gesture_recognizer.getState(button):
                estimate.hand_gesture_recognizer.setState(button, True)
                #print("Pless")
                mouse.press(button=button)
        else:
            if estimate.hand_gesture_recognizer.getState(button):
                estimate.hand_gesture_recognizer.setState(button, False)
                #print("Release")
                mouse.release(button=button)

    

class Config:
    def __init__(self, *, cursor_size:int=10, lefty:bool=False, pointer_offset:Vector2D=Vector2D(x=350, y=100), screen_size:Vector2D, fine_mouse_sensitivity:float=0.22):
        self.cursor_size = cursor_size
        self.lefty = lefty
        self.pointer_offset = pointer_offset
        self.screen_size = screen_size
        self.fine_mouse_sensitivity = fine_mouse_sensitivity
screen = get_monitors()[0]
config = Config(screen_size=Vector2D(x=screen.width, y=screen.height))
smoothing = Smoothing2D(10)
estimate = Estimate(config)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    image = estimate.update(image)
    #print(estimate.hand_gesture_recognizer.isMousePressed().right["left"])
    #if estimate.hand_gesture_recognizer.isMousePressed().right["left"]:
        #if not mouse.is_pressed(button='left'):
            #mouse.press(button='left')
    #else:
        #if mouse.is_pressed(button='left'):
            #mouse.release(button='left')

    mouseController(config, estimate=estimate, smoothing=smoothing)


    """
    if positions and positions.left and estimate.hand_gesture_recognizer.isMouseMoving().left:
        #print(positions.left)
        position = Vector2D(x=positions.left.x*config.screen_size.x, y=positions.left.y*config.screen_size.y)
        smoothing.update(position)
        smoothed = smoothing.get()
        mouse.move(smoothed.x, smoothed.y, True)
    elif positions and positions.right and estimate.hand_gesture_recognizer.isMouseMoving().right:
        #print(positions.right)
        position = Vector2D(x=positions.right.x*config.screen_size.x, y=positions.right.y*config.screen_size.y)
        smoothing.update(position)
        smoothed = smoothing.get()
        mouse.move(smoothed.x, smoothed.y, True)
    """
    image = cv2.resize(image, dsize=None, fx=2, fy=2)
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    #print(time.time()-t)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()


