import os
import numpy as np
 
DATA_PATH = os.path.join('dataset')
CLASSES = np.array(['baby_crawling', 'baby_walking', 'baby_still'])
NO_SEQUENCES = 300
SEQUENCE_LEN = 10