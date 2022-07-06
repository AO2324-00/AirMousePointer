import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

import numpy as np
from matplotlib import pyplot as plt
import vector
import screen

#print(mp_drawing_styles.get_default_pose_landmarks_style())

'''
入力画像上にランドマークを重ねた画像を生成する関数。
Arguments
 - image:     入力画像
 - landmarks: ランドマークの推定結果
Return
 - ランドマークを重ねた画像
'''
def draw_hands_landmarks(image, landmarks):
    annotated_image = image.copy()
    for hand_landmarks in landmarks.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    return annotated_image

def draw_pose_landmarks(image, landmarks):
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        landmarks.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    return annotated_image


'''
=== 画像の場合 ================================================
'''
IMAGE_FILES = ["./calibration0.png","./calibration1.png"] # 画像のファイルパスを配列に格納して下さい。

hands =  mp_hands.Hands(
    static_image_mode = True,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    max_num_hands = 2,             # 認識する手の最大数。
    model_complexity = 1,          # 手のランドマークモデルの複雑さ(0 or 1)。
    min_detection_confidence = 0.5 # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
)
pose = mp_pose.Pose(
    static_image_mode = True,       # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
    model_complexity = 2,           # 姿勢のランドマークモデルの複雑さ(0, 1 or 2)。
    enable_segmentation = True,     # 姿勢のランドマークに加えて、セグメンテーションマスク(背景のマスク)を生成するか。
    min_detection_confidence = 0.5  # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
)
# IMAGE_FILESの画像を一枚ずつ処理します。
for index, file in enumerate(IMAGE_FILES):

    # MediaPipeHandsでは、入力画像は左右反転したものであると仮定して処理されます。
    # その対策として、事前に入力画像を左右反転処理を行います。
    image = cv2.flip(cv2.imread(file), 1) 

    # BGR画像をRGBに変換します。
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 画像からランドマークを推定します。
    hands_landmarks = hands.process(image)
    pose_landmarks = pose.process(image)

    # ランドマークが推定できていない場合はスキップします。
    if not hands_landmarks.multi_hand_landmarks or not pose_landmarks.pose_landmarks or len(hands_landmarks.multi_hand_landmarks) < 2:
        continue

    # 画像上に推定したランドマークを描画します。
    annotated_image = draw_hands_landmarks(image, hands_landmarks)
    annotated_image = draw_pose_landmarks(annotated_image, pose_landmarks)

    #p0 = vector.calcMiddleVector(hands_landmarks.multi_hand_landmarks[0].landmark[2], hands_landmarks.multi_hand_landmarks[0].landmark[5])
    #p1 = vector.calcMiddleVector(hands_landmarks.multi_hand_landmarks[1].landmark[2], hands_landmarks.multi_hand_landmarks[1].landmark[5])
    v0 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[0].landmark[5], hands_landmarks.multi_hand_landmarks[0].landmark[8])
    v1 = vector.calcVector3D(hands_landmarks.multi_hand_landmarks[1].landmark[5], hands_landmarks.multi_hand_landmarks[1].landmark[8])
    print('origin: ', hands_landmarks.multi_hand_landmarks[0].landmark[0])
    print(v0, v1)
    v = vector.Vector3D({'x': v0.x-v1.x, 'y': v0.y-v1.y, 'z': v0.z-v1.z})
    print(v)
    p0 = pose_landmarks.pose_landmarks.landmark[19]
    p1 = pose_landmarks.pose_landmarks.landmark[20]
    s0 = pose_landmarks.pose_landmarks.landmark[11]
    s1 = pose_landmarks.pose_landmarks.landmark[12]
    e0 = pose_landmarks.pose_landmarks.landmark[2]
    e1 = pose_landmarks.pose_landmarks.landmark[5]
    #v = vector.calcVector3D(s0, s1)
    screen_vertex = screen.calc_vertex(v, p0, p1)
    annotated_image = screen.draw_border(annotated_image, screen_vertex)

    # 左右の反転を元に戻します。
    annotated_image = cv2.flip(annotated_image, 1)

    # RGB画像をBGRに変換します。
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    #mp_drawing.plot_landmarks(pose_landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    mp_drawing.plot_landmarks(hands_landmarks.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    #mp_drawing.plot_landmarks(hands_landmarks.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
    screen.plot_vertex(screen_vertex,
        pose_landmarks.pose_landmarks.landmark[2], pose_landmarks.pose_landmarks.landmark[5],
        pose_landmarks.pose_landmarks.landmark[11], pose_landmarks.pose_landmarks.landmark[12])

    # ランドマークを描画した画像を出力します。
    cv2.imwrite('./annotated_image_' + str(index) + '.png', annotated_image)

