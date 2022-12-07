import sys
sys.path.append('../')

import cv2
import mediapipe as mp
import numpy as np

from Commons.vector import calcVector2D, Vector2D, Vector3D, calcDotProduct, calcMiddleVector
from Commons.landmarks import LandmarkPoint

hands =  mp.solutions.hands.Hands(
    static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    max_num_hands = 4,              # 認識する手の最大数。
    model_complexity = 1,           # 手のランドマークモデルの複雑さ(0 or 1)。
    #min_detection_confidence = 0.3,
    min_detection_confidence = 0.4, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
    min_tracking_confidence = 0.4   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
)
pose = mp.solutions.pose.Pose(
    static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    model_complexity = 2,           # ランドマークモデルの複雑さ(0 or 1 or 2)。
    min_detection_confidence = 0.5, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
    min_tracking_confidence = 0.5   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
)

def userRecognition(image, *, screen=None, max=5, tracking=False):

    original_image = image.copy()
    original_image.flags.writeable = False
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    masked_image = original_image.copy()
    user_box = None
    Pose = None
    count = 0
    if tracking:
        Pose = pose.process(masked_image)
        if Pose and Pose.pose_landmarks and checkUser(screen, Pose):
            masked_image, user_box = getMaskedImage(masked_image, Pose, isInside=False, original_image=original_image)
        else:
            tracking = False
    if not tracking:
        while count <= max:
            count += 1
            Pose = pose.process(masked_image)
            if not (Pose and Pose.pose_landmarks):
                return None, None, user_box, tracking
            if checkUser(screen, Pose):
                masked_image, user_box = getMaskedImage(masked_image, Pose, isInside=False, original_image=original_image)
                tracking = True
                break
            else:
                masked_image, user_box = getMaskedImage(masked_image, Pose, isInside=True, original_image=original_image)
    Hands = hands.process(masked_image)

    return Pose, Hands, user_box, tracking

def getMaskedImage(image, Pose, *, isInside: bool, original_image):
    pose_landmark = [[lmk.x, lmk.y, lmk.z] for lmk in Pose.pose_landmarks.landmark]

    # 出力される座標は0〜1なので，画像の座標に揃える
    frame_height, frame_width = image.shape[:2]
    pose_landmark *= np.array([frame_width, frame_height, frame_width])

    x = [int(lmk[0]) for lmk in pose_landmark]
    y = [int(lmk[1]) for lmk in pose_landmark]
    min_x, min_y = min(x), min(y)
    max_x, max_y = max(x), max(y)
    margin = int(min(max_x-min_x, max_y-min_y) / 8)
    min_x, min_y = max(min_x-margin*1.5,0), max(min_y-margin*2,0)
    max_x, max_y = min(max_x+margin*1.5,frame_width-1), min(max_y+margin*2,frame_height-1)
    min_x, min_y = int(min_x), int(min_y)
    max_x, max_y = int(max_x), int(max_y)
    masked_image = image.copy()
    if isInside:
        cv2.rectangle(masked_image, (min_x, min_y), (max_x, max_y), (0, 0, 0), thickness=-1)
        return masked_image, None
    else:
        mask = np.zeros((frame_height, frame_width,3), np.uint8)
        mask = cv2.rectangle(mask, (min_x,min_y),(max_x,max_y),(255,255,255), -1)
        masked_image = cv2.bitwise_and(original_image, mask)
        height, width, _ = original_image.shape
        scale = Vector2D(x=width, y=height)
        min_x, min_y = min_x/scale.x, min_y/scale.y
        max_x, max_y = max_x/scale.x, max_y/scale.y
        return masked_image, np.array([
            LandmarkPoint(Vector3D(x=min_x, y=min_y, z=0), scale=scale),
            LandmarkPoint(Vector3D(x=max_x, y=min_y, z=0), scale=scale),
            LandmarkPoint(Vector3D(x=max_x, y=max_y, z=0), scale=scale),
            LandmarkPoint(Vector3D(x=min_x, y=max_y, z=0), scale=scale)
        ])

def checkUser(screen, Pose):
    left_arm = Vector2D.fromVector(Pose.pose_landmarks.landmark[15])
    right_arm = Vector2D.fromVector(Pose.pose_landmarks.landmark[16])
    body_top = calcMiddleVector(Vector2D.fromVector(Pose.pose_landmarks.landmark[11]), Vector2D.fromVector(Pose.pose_landmarks.landmark[12]))
    body_bottom = calcMiddleVector(Vector2D.fromVector(Pose.pose_landmarks.landmark[23]), Vector2D.fromVector(Pose.pose_landmarks.landmark[24]))
    body_point = calcMiddleVector(body_top, body_bottom)
    body_point = calcMiddleVector(body_point, body_bottom)
    if not screen:
        return left_arm.y < body_point.y and right_arm.y < body_point.y
    return left_arm.y < body_point.y or right_arm.y < body_point.y