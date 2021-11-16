import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import mediapipe as mp
import constants as c
from tensorflow.keras.models import load_model
import utils as util
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('model_baby_99acc_22-8_4.h5')
    
def negate(x):
    return x * -1

def is_video(file):
    if file.type.split('/')[0] == 'video':
        return True
    else:
        return False

def main():
    st.header("Baby Gesture Detection")
    st.write("You can find sample media for testing [here](https://github.com/shreyas-jk/Baby-Action-Detection-Safety-System-Prototype/tree/main/streamlit/sample)")
    st.markdown("""---""")
    file_object = st.file_uploader("Upload Media", type=["mp4"])

    if file_object is not None and is_video(file_object):
        class_list = []
        sequence = []
    
        # Save file
        destination = os.path.join("uploads", file_object.name)
        with open(destination, "wb") as f:
            f.write((file_object).getbuffer())

        cap = cv2.VideoCapture(destination)
        i = 0
        t = st.empty()
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                i += 1
                t.markdown('Please wait.. Processing frame {0}'.format(i))
                
                # Read feed
                ret, frame = cap.read()

                # Make detections
                try:
                    image, results = util.mediapipe_detection(frame, holistic)
                except:
                    break
                
                # Draw landmarks
                util.draw_styled_landmarks(mp_holistic, mp_drawing, image, results)
                
                # Add keypoints
                keypoints = util.extract_keypoints(results)
                sequence.append(keypoints)
            
                # Select last SEQUENCE_LEN sequences only
                sequence = sequence[negate(c.SEQUENCE_LEN):]

                if len(sequence) == c.SEQUENCE_LEN:         #30
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    class_list.append(c.CLASSES[np.argmax(res)])
            
                # Exit
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        df = pd.DataFrame({'classes': class_list})
        df = df.groupby('classes')["classes"].size().reset_index(name='count')
        total_count = df['count'].sum()
        
        t.empty()

        # Print results
        for index, row in df.iterrows():
            st.write('Baby was {0} for {1}% of time'.format(row['classes'].replace('baby_', ''),round((row['count']/total_count)*100, 2) ))

        # Show pie
        colors = sns.color_palette('pastel')[0:5]
        fig, ax = plt.subplots()
        ax.pie(list(df['count']), labels = np.unique(class_list), colors=colors, autopct='%.0f%%')
        st.pyplot(fig)

if __name__ == "__main__":
    main()