import cv2
import mediapipe as mp

class Draw:

    @staticmethod
    def point(image, point_landmark, *, color=(150, 245, 250), markerType=cv2.MARKER_TRIANGLE_UP, thickness=2, markerSize=10):
        cv2.drawMarker(image, (int(point_landmark.raw_landmark.x), int(point_landmark.raw_landmark.y)), color=color, markerType=markerType, thickness=thickness, markerSize=markerSize)
        return image
    
    @staticmethod
    def line(image, origin_landmarks, point_landmark, *, color=(255, 255, 255), thickness=1):
        cv2.line(image, (int(origin_landmarks.raw_landmark.x), int(origin_landmarks.raw_landmark.y)), (int(point_landmark.raw_landmark.x), int(point_landmark.raw_landmark.y)), color, thickness)
        return image

    @staticmethod
    def hands(image, landmarks, *, color=(15, 50, 255), thickness=1):
        for dir in ["left", "right"]:
            if not landmarks or not landmarks.get(dir):
                continue
            for index in mp.solutions.hands.HAND_CONNECTIONS:
                cv2.line(image, (int(landmarks.get(dir).raw_landmark[index[0]].x), int(landmarks.get(dir).raw_landmark[index[0]].y)), (int(landmarks.get(dir).raw_landmark[index[1]].x), int(landmarks.get(dir).raw_landmark[index[1]].y)), color, thickness)
        return image

    @staticmethod
    def pose(image, landmarks, *, color=(100, 110, 104), thickness=1):
        if not landmarks:
            return image
        for index in mp.solutions.pose.POSE_CONNECTIONS:
            cv2.line(image, (int(landmarks.raw_landmark[index[0]].x), int(landmarks.raw_landmark[index[0]].y)), (int(landmarks.raw_landmark[index[1]].x), int(landmarks.raw_landmark[index[1]].y)), color, thickness)
        return image

    @staticmethod
    def screenBox(image, eye_landmark, screen_vertex, *, color=(255, 255, 255), thickness=1):
        for index in range(len(screen_vertex)):
            cv2.line(image, (int(eye_landmark.raw_landmark.x), int(eye_landmark.raw_landmark.y)), (int(screen_vertex[index].raw_landmark.x), int(screen_vertex[index].raw_landmark.y)), color, thickness)
            cv2.line(image, (int(screen_vertex[index-1].raw_landmark.x), int(screen_vertex[index-1].raw_landmark.y)), (int(screen_vertex[index].raw_landmark.x), int(screen_vertex[index].raw_landmark.y)), color, thickness)
        return image

    @staticmethod
    def screenPanel(image, screen_vertex, *, color=(255, 255, 255), thickness=1):
        for index in range(len(screen_vertex)):
            cv2.line(image, (int(screen_vertex[index-1].raw_landmark.x), int(screen_vertex[index-1].raw_landmark.y)), (int(screen_vertex[index].raw_landmark.x), int(screen_vertex[index].raw_landmark.y)), color, thickness)
        return image