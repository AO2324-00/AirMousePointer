import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


import vector


def draw_landmark(image, landmarks):
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

# For webcam input:
averaged_landmarks_0 = []
averaged_landmarks_1 = []
for i in range(21):
    averaged_landmarks_0.append(vector.Average3D(8, 1, 1000))
    averaged_landmarks_1.append(vector.Average3D(8, 1, 1000))
averaged_landmarks = [averaged_landmarks_0, averaged_landmarks_1]

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./example.mp4')
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    #動画サイズ取得
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  #フレームレート取得
  fps = cap.get(cv2.CAP_PROP_FPS)

  #フォーマット指定
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  #注）グレースケールの画像を出力する場合は第5引数に0を与える
  writer = cv2.VideoWriter('./result.mp4', fmt, fps, (width, height))
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for idk in range(2):
            #print(len(results.multi_hand_landmarks))
            if(len(results.multi_hand_landmarks) < idk+1):
                break
            hand = idk
            if len(averaged_landmarks[0][0].vector) == 0 or len(averaged_landmarks[1][0].vector) == 0:
                hand = idk
            elif(np.abs(results.multi_hand_landmarks[idk].landmark[0].x-averaged_landmarks[0][0].vector[-1][0]) > np.abs(results.multi_hand_landmarks[idk].landmark[0].x-averaged_landmarks[1][0].vector[-1][0])):
                hand = 1
            else:
                hand = 0
            for i in range(21):
                averaged_landmarks[hand][i].update(results.multi_hand_landmarks[idk].landmark[i])
                tmp = averaged_landmarks[hand][i].get()
                results.multi_hand_landmarks[idk].landmark[i].x = tmp.x
                results.multi_hand_landmarks[idk].landmark[i].y = tmp.y
                results.multi_hand_landmarks[idk].landmark[i].z = tmp.z
            
            draw_landmark(image, results.multi_hand_landmarks[idk])
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    #動画書き込み
    writer.write(image)
cap.release()
"""
        for i in range(33):
            averaged_landmarks[i].update(results.pose_landmarks.landmark[i])
            tmp = averaged_landmarks[i].get()
            results.pose_landmarks.landmark[i].x = tmp.x
            results.pose_landmarks.landmark[i].y = tmp.y
            results.pose_landmarks.landmark[i].z = tmp.z
        """