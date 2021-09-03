import cv2
import numpy as np
import mediapipe as mp
import constants as c

BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB 
    image.flags.writeable = False                  # Image is no longer writeable
    results = holistic.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints_js(results):
    pose = np.array([[res.get("x"), res.get("y"), res.get("z"), res.get("visibility")] for res in results.get("poseLandmarks")]).flatten() if results.get("poseLandmarks") else np.zeros(33*4)
    face = np.array([[res.get("x"), res.get("y"), res.get("z")] for res in results.get("faceLandmarks")]).flatten() if results.get("faceLandmarks") else np.zeros(468*3)
    lh = np.array([[res.get("x"), res.get("y"), res.get("z")] for res in results.get("leftHandLandmarks")]).flatten() if results.get("leftHandLandmarks") else np.zeros(21*3)
    rh = np.array([[res.get("x"), res.get("y"), res.get("z")] for res in results.get("rightHandLandmarks")]).flatten() if results.get("rightHandLandmarks") else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    
def draw_styled_landmarks(mp_holistic, mp_drawing, image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=WHITE_COLOR, thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=WHITE_COLOR, thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=1)
                             )
     