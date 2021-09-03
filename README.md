
# Baby Action Detection

## Table of Contents

- [About](#about)
- [Install](#installing)
- [Run Predictions](#run-predictions)
- [Demo](#demo)

# About 

An attempt to harness the power of Deep Learning to come up with a solution that can let us detect various classes of activities an infant, toddler or a baby is performing in real-time. This POC can then be published as an end-to-end deployable cloud project.

The model does not restrict predictions for babies only, it is applicable to all entities that appears in a human posture. So temporary, this needs to be handled at project level.

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
