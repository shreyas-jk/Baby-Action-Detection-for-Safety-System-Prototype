
# Baby Action Detection

## Table of Contents

- [About](#about)
- [Install](#install)
- [Run Predictions](#run-predictions)
- [Demo](#demo)

# About 

An attempt to harness the power of Deep Learning to come up with a solution that can let us detect various classes of activities an infant, toddler or a baby is performing in real-time. This POC can then be published as an end-to-end deployable cloud project.

The model does not restrict predictions for babies only, it is applicable to all entities that appears in a human posture. So temporary, this needs to be handled at project level.

 Special thanks to [nicknochnack/ActionDetectionforSignLanguage](#https://github.com/nicknochnack/ActionDetectionforSignLanguage) repository for putting up such helpful content. Without it this project might have never existed.

## Data collection
Data was collected from YouTube video clips. Human pose keypoints were extracted with the help of [MediaPipe](#https://mediapipe.dev/).

# Install

Create a new environment and use below command for installing all required packages

```bash
pip install -r- requirements.txt
```

# Run Predictions

1. Rename your baby video as input.mp4 and place it inside ```/raw``` directory.
2. Open cmd and traverse to the project directory.
3. To run the prediction script, just do:

```bash
python prediction.py 
```

# Demo

![GIF](https://github.com/shreyas-jk/Baby-Action-Detection_Safety-System-Prototype/blob/main/demo/1.gif?raw=true)

![GIF](https://github.com/shreyas-jk/Baby-Action-Detection_Safety-System-Prototype/blob/main/demo/2.gif?raw=true)

![GIF](https://github.com/shreyas-jk/Baby-Action-Detection_Safety-System-Prototype/blob/main/demo/3.gif?raw=true)