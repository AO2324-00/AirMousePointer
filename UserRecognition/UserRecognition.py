#from ..Commons.landmarks import Landmarks

import cv2
import mediapipe as mp
import numpy as np

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

def userRecognition(image, screen=None, max=10):

    masked_image = image.copy()
    masked_image.flags.writeable = False
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    Pose = None
    count = 0
    while count <= max:
        count += 1
        Pose = pose.process(masked_image)
        if not Pose.pose_landmarks:
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            return None, None, masked_image
        isUser = True
        if isUser:
            masked_image = getMaskedImage(masked_image, Pose, isInside=False, original_image=image)
            break
        else:
            masked_image = getMaskedImage(masked_image, Pose, isInside=True)
    Hands = hands.process(masked_image)

    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    return Pose, Hands, masked_image

def getMaskedImage(image, Pose, *, isInside: bool, original_image=None):
    pose_landmark = [[lmk.x, lmk.y, lmk.z] for lmk in Pose.pose_landmarks.landmark]

    # 出力される座標は0〜1なので，画像の座標に揃える
    frame_height, frame_width = image.shape[:2]
    pose_landmark *= np.array([frame_width, frame_height, frame_width])

    x = [int(lmk[0]) for lmk in pose_landmark]
    y = [int(lmk[1]) for lmk in pose_landmark]
    min_x, min_y = min(x), min(y)
    max_x, max_y = max(x), max(y)
    margin = int(min(max_x-min_x, max_y-min_y) / 8)
    min_x, min_y = max(min_x-margin,0), max(min_y-margin*2,0)
    max_x, max_y = min(max_x+margin,frame_width-1), min(max_y+margin*2,frame_height-1)
    masked_image = image.copy()
    if isInside:
        cv2.rectangle(masked_image, (min_x, min_y), (max_x, max_y), (0, 0, 0), thickness=-1)
    else:
        mask = np.zeros((frame_height, frame_width,3), np.uint8)
        mask = cv2.rectangle(mask, (min_x,min_y),(max_x,max_y),(255,255,255), -1)
        masked_image = cv2.bitwise_and(original_image.copy() if original_image else masked_image, mask)

    return masked_image
