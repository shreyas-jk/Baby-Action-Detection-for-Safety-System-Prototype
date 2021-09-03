import cv2
import numpy as np
import mediapipe as mp
import constants as c
from tensorflow.keras.models import load_model
import utils as util
from apscheduler.schedulers.background import BackgroundScheduler
import time

#Declarations
prev_result_time = time.time()
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('model/model_baby_99acc_22-8_4.h5')
sequence = []
                        
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)
mbox_w_2, mbox_h_2 = 180, 420
gl_activity_pred = "Please wait"

# Prediction Function
def predict_activity():
    if len(sequence) == c.SEQUENCE_LEN:         #30
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        global gl_activity_pred, prev_result_time
        gl_activity_pred = c.CLASSES[np.argmax(res)]
        prev_result_time = time.time()

# Convert positive number to negative
def negate(x):
    return x * -1

# Prediction runs every secq
sched = BackgroundScheduler()
sched.add_job(predict_activity, 'interval', seconds=1)
sched.start()


# cap = cv2.VideoCapture('https://192.168.29.176:8080/video') 
cap = cv2.VideoCapture('raw/input.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output/video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = util.mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        util.draw_styled_landmarks(mp_holistic, mp_drawing, image, results)
    
        # Add keypoints
        keypoints = util.extract_keypoints(results)
        sequence.append(keypoints)
    
        # Select last SEQUENCE_LEN sequences only
        sequence = sequence[negate(c.SEQUENCE_LEN):]
        
        cv2.rectangle(image, (30, 10), (300, 70), (0, 0, 0), -1)
        cv2.putText(image, gl_activity_pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display
        cv2.imshow('Frame', image)

        # write to file
        out.write(image)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()