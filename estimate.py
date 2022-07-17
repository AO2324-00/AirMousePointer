import numpy as np
from numpy import linalg as LA
import cv2
import mediapipe as mp

import vector
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
        self.screen = None

        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        
        self.hands =  mp_hands.Hands(
            static_image_mode = False,      # 単体の画像かどうか(Falseの場合は入力画像を連続したものとして扱います)。
            max_num_hands = 2,              # 認識する手の最大数。
            model_complexity = 1,           # 手のランドマークモデルの複雑さ(0 or 1)。
            min_detection_confidence = 0.3, # 検出が成功したと見なされるための最小信頼値(0.0 ~ 1.0)。
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

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = adjust(image, 1.7, 30.0)

        hands_landmarks = self.hands.process(image)
        pose_landmarks = self.pose.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if pose_landmarks.pose_landmarks:
            image = image
            image = draw_pose_landmarks(image, pose_landmarks)

        #print(hands_landmarks.multi_hand_landmarks)
        if pose_landmarks.pose_landmarks:# and hands_landmarks.multi_hand_landmarks:
            if hands_landmarks.multi_hand_landmarks:
                height, width, _ = image.shape
                print(vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[2], pose_landmarks.pose_landmarks.landmark[5], vector.Vector3D({'x': width, 'y': height, 'z': width}))/width,
                    (vector.calcDistance3D(hands_landmarks.multi_hand_landmarks[0].landmark[5], hands_landmarks.multi_hand_landmarks[0].landmark[17], vector.Vector3D({'x': width, 'y': height, 'z': width}))+vector.calcDistance3D(hands_landmarks.multi_hand_landmarks[0].landmark[0], hands_landmarks.multi_hand_landmarks[0].landmark[9], vector.Vector3D({'x': width, 'y': height, 'z': width})))/(width*2))
                image = draw_hands_landmarks(image, hands_landmarks)
            # キャリブレーション
            screen_vertex, state, screen_offsets = self.calibration.calcPosition(image, hands_landmarks, pose_landmarks)
            #print(state)
            height, width, _ = image.shape
            left_offset = vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[11], pose_landmarks.pose_landmarks.landmark[13], vector.Vector3D({'x': width, 'y': height, 'z': width}))+vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[13], pose_landmarks.pose_landmarks.landmark[15], vector.Vector3D({'x': width, 'y': height, 'z': width}))
            right_offset = vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[12], pose_landmarks.pose_landmarks.landmark[14], vector.Vector3D({'x': width, 'y': height, 'z': width}))+vector.calcDistance3D(pose_landmarks.pose_landmarks.landmark[14], pose_landmarks.pose_landmarks.landmark[16], vector.Vector3D({'x': width, 'y': height, 'z': width}))
            #print('{:6>.2f}'.format(left_offset), '{:6>.2f}'.format(right_offset))
            if screen_vertex != None:
                image = screen.draw_border(image, screen_vertex, (255, 255, 100))
                if state >= 2:
                    #self.screen = screen.SpatialPlane(screen_vertex, (screen_offsets['left']+screen_offsets['right'])/2)
                    self.screen = screen.SpatialPlane(screen_vertex, vector.calcMiddleVector(pose_landmarks.pose_landmarks.landmark[11], pose_landmarks.pose_landmarks.landmark[12]).z, 1)
            elif self.screen != None:
                #self.screen.equation.adjustment(max(left_offset, right_offset))
                #self.screen.equation.adjustment((left_offset+right_offset)/2)
                self.screen.equation.adjustment(0)
                image = screen.draw_border(image, self.screen.getVertex())

                """
                self.screen.equation.adjustment(left_offset)
                point0 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(pose_landmarks.pose_landmarks.landmark[19])
                image = screen.draw_line(image, [point0, pose_landmarks.pose_landmarks.landmark[19]], color=(255, 0, 0))
                image = screen.draw_point(image, point0, (255, 0, 0))
                self.screen.equation.adjustment(right_offset)
                point1 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(pose_landmarks.pose_landmarks.landmark[20])
                image = screen.draw_line(image, [point1, pose_landmarks.pose_landmarks.landmark[20]], color=(0, 0, 255))
                image = screen.draw_point(image, point1, (0, 0, 255))
               
                """
                """
                mid0 = vector.calcMiddleVector(pose_landmarks.pose_landmarks.landmark[11], pose_landmarks.pose_landmarks.landmark[12])
                mid = vector.calcMiddleVector(pose_landmarks.pose_landmarks.landmark[2], pose_landmarks.pose_landmarks.landmark[5])
                #mid.z = mid0.z
                #mid.z = 0
                mid = vector.Vector3D({'x': mid.x, 'y': mid.y, 'z': mid0.z + (mid.z - mid0.z)*0.5})
                image = screen.draw_point(image, mid, (0, 255, 0))

                self.screen.equation.adjustment()
                point0 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(mid, vector.calcVector3D(mid, pose_landmarks.pose_landmarks.landmark[19]))
                image = screen.draw_point(image, point0, (0, 255, 0))
                #self.screen.equation.adjustment(left_offset)
                self.screen.equation.adjustment(pose_landmarks.pose_landmarks.landmark[11].z)
                #print('{:6>.2f}'.format(self.screen.equation.offset))
                point0 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(mid, vector.calcVector3D(mid, pose_landmarks.pose_landmarks.landmark[19]))
                image = screen.draw_line(image, [mid, pose_landmarks.pose_landmarks.landmark[19]], color=(255, 0, 0))
                image = screen.draw_point(image, point0, (255, 0, 0))
                position_0 = screen.calc_position(self.screen.getVertex(), point0)

                self.screen.equation.adjustment()
                point1 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(mid, vector.calcVector3D(mid, pose_landmarks.pose_landmarks.landmark[20]))
                image = screen.draw_point(image, point1, (0, 255, 0))
                #self.screen.equation.adjustment(right_offset)
                self.screen.equation.adjustment(pose_landmarks.pose_landmarks.landmark[12].z)
                point1 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(mid, vector.calcVector3D(mid, pose_landmarks.pose_landmarks.landmark[20]))
                image = screen.draw_line(image, [mid, pose_landmarks.pose_landmarks.landmark[20]], color=(0, 0, 255))
                image = screen.draw_point(image, point1, (0, 0, 255))
                position_1 = screen.calc_position(self.screen.getVertex(), point1)
                #print(position_0, position_1)
                """
                point0 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(pose_landmarks.pose_landmarks.landmark[19])
                position_0 = screen.calc_position(self.screen.getVertex(), point0)
                image = screen.draw_point(image, point0, (255, 0, 0))
                point1 = screen.SpatialPlane(self.screen.getVertex()).calcIntersection(pose_landmarks.pose_landmarks.landmark[20])
                position_1 = screen.calc_position(self.screen.getVertex(), point1)
                #print(position_0, position_1)
                cv2.imshow('Screen', screen.draw_screen([position_0, position_1]))
                image = screen.draw_point(image, point1, (0, 0, 255))
                #self.plot.update(self.screen, pose_landmarks, mid, [point0, point1])
                #self.plot.update(self.screen, pose_landmarks, [point0, point1])
                #self.plot.update(self.screen, pose_landmarks, pose_landmarks.pose_landmarks.landmark[20], [point1, point1])
        else:
            cv2.imshow('Screen', screen.draw_screen([]))
            #self.plot.update(self.screen, pose_landmarks)
            if self.screen != None:
                self.screen.equation.adjustment()
                image = screen.draw_border(image, self.screen.getVertex())
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
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