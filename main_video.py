import cv2

import estimate

pointer = estimate.Pointer()
cap = cv2.VideoCapture('./test_video/test.mp4')
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
    pointer.process(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()