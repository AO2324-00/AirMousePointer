import time

import cv2
import pyautogui
import mouse

import estimate
import vector

cursor_size = 10
lefty=False
offset=vector.Vector2D({'x': 400, 'y': 100})

width, height = pyautogui.size()
pointer = estimate.Pointer()
cap = cv2.VideoCapture(0)

def calcPosition(positions, offset=vector.Vector2D({'x': 0, 'y': 0})):
    if not (positions.left and positions.right):
        return None
    if positions.left:
        positions.left.x += offset.x/width
        positions.left.y -= offset.y/height
    if positions.right:
        positions.right.x -= offset.x/width
        positions.right.y -= offset.y/height

    for side in ['left', 'right']:
        position = positions.get(side)
        if not position:
            continue
        for dir in ['x', 'y']:
            if position.get(dir) < -1:
                    positions.set(side, None)
                    break
            elif position.get(dir) < 0:
                position.set(dir, 0)
            elif 2 < position.get(dir):
                positions.set(side, None)
                break
            elif 1 < position.get(dir):
                position.set(dir, 1)
    #print(position)
    positions = [positions.left, positions.right]
    #print(position)
    positions = [side for side in positions if side is not None]
    if len(positions) > 0:
        positions = positions[0 if lefty else 1] if len(positions) == 2 else positions[0]
        x, y = (int(positions.x*width), int(positions.y*height))
        x -= cursor_size if x > width-cursor_size else 0
        y -= cursor_size if y > height-cursor_size else 0
        return vector.Vector2D({'x': x, 'y': y})
    return None
smoothing = vector.Smoothing2D(13)
while cap.isOpened():
    t = time.time()
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    #print(time.time()-t)
    #t = time.time()
    
    image, positions = pointer.process(image)

    position = calcPosition(positions, offset)
    if position:
        print(position)
        smoothing.update(position)
        p = smoothing.get()
        mouse.move(p.x, p.y, True)
        #print(time.time()-t)

    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    #print(time.time()-t)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()


