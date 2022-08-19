import numpy as np
import cv2
import mediapipe as mp

from hand_gesture_recognizer import HandGestureRecognizer
from screen import RealtimePlot, draw_border, draw_line, draw_point

from landmarks import Landmarks

from virtual_screen import VirtualScreen
from pointer import Pointer

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
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

class Estimate:

    def __init__(self, config):

        self.__config = config

        self.__hands =  mp.solutions.hands.Hands(
            static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
            max_num_hands = 2,              # 認識する手の最大数。
            model_complexity = 1,           # 手のランドマークモデルの複雑さ(0 or 1)。
            #min_detection_confidence = 0.3,
            min_detection_confidence = 0.7, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
            min_tracking_confidence = 0.6   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
        )
        self.__pose = mp.solutions.pose.Pose(
            static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
            model_complexity = 2,           # ランドマークモデルの複雑さ(0 or 1 or 2)。
            min_detection_confidence = 0.5, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
            min_tracking_confidence = 0.5   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
        )

        self.landmarks: Landmarks = Landmarks(eye=None, hands=None)
        self.virtual_screen = VirtualScreen()
        self.pointer = Pointer(config=self.__config)
        self.hand_gesture_recognizer = HandGestureRecognizer()

        self.plot = RealtimePlot()

    def __del__(self):
        self.__hands.close()
        self.__pose.close()

    def update(self, image: np.ndarray):

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = adjust(image, 1.7, 30.0)

        Hands = self.__hands.process(image)
        Pose = self.__pose.process(image)
        self.landmarks = Landmarks.parse(image, Pose=Pose, Hands=Hands, hand_index=range(21)).update(self.landmarks)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.virtual_screen.update(self.landmarks)
        self.hand_gesture_recognizer.update(self.landmarks)
        #print(self.hand_gesture_recognizer.isMouseMoving())
        self.pointer.calcPosition(self.virtual_screen.getSpatialPlane(), self.landmarks)

        if self.virtual_screen.getSpatialPlane():
            tmp = self.pointer.getPoints()
            #self.plot.update(self.virtual_screen.getSpatialPlane().getVertex(), landmarks=self.landmarks, target_landmarks=[tmp.left, tmp.right])
            image = draw_border(image, self.virtual_screen.getSpatialPlane().getVertex())
            if tmp.left:
                image = draw_point(image, tmp.left, color=(50, 50, 255))
                image = draw_line(image, [self.landmarks.eye, tmp.left], color=(50, 50, 255))
            if tmp.right:
                image = draw_point(image, tmp.right, color=(255, 50, 50))
                image = draw_line(image, [self.landmarks.eye, tmp.right], color=(255, 50, 50))

        if Hands.multi_hand_landmarks:
            image = draw_hands_landmarks(image, Hands)
        if Pose.pose_world_landmarks:
            image = draw_pose_landmarks(image, Pose)

        return image

        



def adjustBrightness(image):
    b1,g1,r1 = cv2.split(image)
    b2 = cv2.equalizeHist(b1)
    g2 = cv2.equalizeHist(g1)
    r2 = cv2.equalizeHist(r1)
    adjusted_image = cv2.merge((b2,g2,r2))
    return adjusted_image

def adjust(image, alpha=1.0, beta=0.0):
    # 積和演算を行う。
    dst = alpha * image + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)