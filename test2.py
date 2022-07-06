import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import vector

# For webcam input:
cap = cv2.VideoCapture(0)
body_average = vector.Average2D(8, 1)
landmark_12 = vector.Average3D(8, 1)
landmark_11 = vector.Average3D(8, 1)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    height, width, _ = image.shape
    if results.pose_landmarks:
        #print(vector.calcDistance3D(results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[14]),vector.calcDistance3D(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[13]))
        #angleL = vector.calcAngle(results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[11], vector.Vector3D({'x': width, 'y': height, 'z': width}))
        #print(angleL)
        #angleBody = vector.calcAngle(results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[11])
        #body_average.update(angleBody)
        landmark_12.update(results.pose_landmarks.landmark[12])
        landmark_11.update(results.pose_landmarks.landmark[11])
        angleBody = vector.calcAngle(landmark_12.get(), landmark_11.get(), vector.Vector3D({'x': width, 'y': height, 'z': width}))
        #print(landmark_11.get())
        print(angleBody)
        #print( body_average.get())
        #body_average.get()
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()