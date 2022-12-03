import sys
sys.path.append('../')

import cv2
import mediapipe as mp
import numpy as np

from Commons.landmarks import Landmarks
from Commons.vector import Vector2D

def adjust(image, alpha=1.0, beta=0.0):
    # 積和演算を行う。
    dst = alpha * image + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)

hands =  mp.solutions.hands.Hands(
    static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    max_num_hands = 2,              # 認識する手の最大数。
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
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
print(mp.solutions.pose.POSE_CONNECTIONS)
def draw_pose_landmarks(image, landmarks):
    annotated_image = image.copy()
    if not landmarks:
        return annotated_image
    height, width, _ = image.shape
    for index in mp.solutions.pose.POSE_CONNECTIONS:
        cv2.line(annotated_image, (int(landmarks.landmark[index[0]].x*width), int(landmarks.landmark[index[0]].y*height)), (int(landmarks.landmark[index[1]].x*width), int(landmarks.landmark[index[1]].y*height)), color=(0,255,0), thickness=2)
    return annotated_image

def draw_hands_landmarks(image, landmarks):
    annotated_image = image.copy()
    height, width, _ = image.shape
    for dir in ["left", "right"]:
        if not landmarks.get(dir):
            continue
        for index in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(annotated_image, (int(landmarks.get(dir).landmark[index[0]].x*width), int(landmarks.get(dir).landmark[index[0]].y*height)), (int(landmarks.get(dir).landmark[index[1]].x*width), int(landmarks.get(dir).landmark[index[1]].y*height)), color=(0,0, 255), thickness=2)
    return annotated_image

landmarks = Landmarks()
#cap = cv2.VideoCapture("./input.mkv")
cap = cv2.VideoCapture(0)
"""
# 動画サイズ取得します。
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# フレームレート取得します。
fps = cap.get(cv2.CAP_PROP_FPS)
# フォーマット指定します。
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# VideoWriterを作成します。
video = cv2.VideoWriter('./output.mp4', fmt, fps, (width, height))
print(width, height, fps, fmt)
"""
while cap.isOpened():
    
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
        #break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = adjust(image, 1.7, 30.0)

    Pose = pose.process(image)
    Hands = hands.process(image)
    height, width, _ = image.shape
    landmarks.update(Pose, Hands, scale=Vector2D(x=width, y=height))

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = draw_pose_landmarks(image, landmarks.pose)
    image = draw_hands_landmarks(image, landmarks.hands)

    #image = cv2.resize(image, dsize=None, fx=2, fy=2)
    cv2.imshow('Landmarks test', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break

    # 動画を書き込みます。
    #video.write(image)
#video.release()
    
cap.release()

