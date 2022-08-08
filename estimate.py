import numpy as np
import cv2
import mediapipe as mp

import vector
import landmarks
import screen
import calibration

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

class Pointer:

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.plot = screen.RealtimePlot()

        self.calibration = calibration.HandState()
        self.LandmarkParser = landmarks.LandmarkParser()
        self.screen = None

        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose

        
        self.hands =  mp_hands.Hands(
            static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
            max_num_hands = 2,              # 認識する手の最大数。
            model_complexity = 1,           # 手のランドマークモデルの複雑さ(0 or 1)。
            #min_detection_confidence = 0.3,
            min_detection_confidence = 0.5, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
            min_tracking_confidence = 0.4   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
        )
        """
        self.hands =  mp_hands.Hands(
            static_image_mode = True,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
            max_num_hands = 2,              # 認識する手の最大数。
        )
        """
        self.pose = mp_pose.Pose(
            static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
            model_complexity = 2,           # ランドマークモデルの複雑さ(0 or 1 or 2)。
            min_detection_confidence = 0.5, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
            min_tracking_confidence = 0.5   # 前のフレームからランドマークが正常に追跡されたとみなされるための最小信頼度(0.0 ~ 1.0)。
        )

    def __del__(self):
        self.hands.close()
        self.pose.close()

    def process(self, image):


        pointer = landmarks.BothSides()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = adjust(image, 1.7, 30.0)

        hands_landmarks = self.hands.process(image)
        pose_landmarks = self.pose.process(image)
        #return image, pointer
        if hands_landmarks.multi_hand_landmarks:
            image = draw_hands_landmarks(image, hands_landmarks)

        results = self.LandmarkParser.update(image, pose_landmarks, hands_landmarks, [1, 2, 4, 5, 8])

        #print(results.eye, results.hands.right)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #print(hands_landmarks.multi_hand_landmarks)
        if results.eye: # and results.hands.left and results.hands.right:# and hands_landmarks.multi_hand_landmarks:

            # キャリブレーション
            screen_vertex, state = self.calibration.calcPosition(results.eye, results.hands)
            #print(screen_vertex)
            #print(state)
            #print('{:6>.2f}'.format(left_offset), '{:6>.2f}'.format(right_offset))
            if screen_vertex != None:
                image = screen.draw_border(image, screen_vertex, (255, 255, 100))
                if state >= 2:
                    #self.screen = screen.SpatialPlane(screen_vertex, (screen_offsets['left']+screen_offsets['right'])/2)
                    self.screen = screen.SpatialPlane(screen_vertex)
            elif self.screen != None:
                
                image = screen.draw_border(image, self.screen.getVertex())

                results = self.LandmarkParser.get([5])
                #print(results.hands.left.landmark[0].x, _results.hands.left.landmark[0].x)

                point = []
                position = []
                if results.eye:
                    image = screen.draw_point(image, results.eye, (0, 255, 0))
                if results.hands.left:
                    point_left = self.screen.calcIntersection(results.eye, vector.calcVector3D(results.eye, results.hands.left.landmark[5]))
                    position_left = screen.calc_position(self.screen.getVertex(), point_left)
                    #position_left.x = (position_left.x - 0.5) * 0.9 + 0.5
                    #position_left.y = (position_left.y - 0.5) * 0.9 + 0.5
                    image = screen.draw_point(image, point_left, (255, 0, 0))
                    point.append(point_left)
                    position.append(position_left)
                    pointer.left = position_left
                if results.hands.right:
                    point_right = self.screen.calcIntersection(results.eye, vector.calcVector3D(results.eye, results.hands.right.landmark[5]))
                    position_right = screen.calc_position(self.screen.getVertex(), point_right)
                    #position_right.x = (position_right.x - 0.5) * 0.9 + 0.5
                    #position_right.y = (position_right.y - 0.5) * 0.9 + 0.5
                    #print(position_0, position_1)
                    image = screen.draw_point(image, point_right, (0, 0, 255))
                    point.append(point_right)
                    position.append(position_right)
                    pointer.right = position_right
                #cv2.imshow('Screen', screen.draw_screen(position))
                #self.plot.update(self.screen.getVertex(), results, point)
                #self.plot.update(self.screen, pose_landmarks, [point0, point1])
                #self.plot.update(self.screen, pose_landmarks, pose_landmarks.pose_landmarks.landmark[20], [point_left, point_right])
        else:
            #cv2.imshow('Screen', screen.draw_screen([]))
            #if self.screen:
                #self.plot.update(self.screen.getVertex())
            if self.screen != None:
                image = screen.draw_border(image, self.screen.getVertex())
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        
        return image, pointer


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