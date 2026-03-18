import numpy as np
from modules.sequence_preprocessor import SequencePreprocessor

pre = SequencePreprocessor()

dummy = np.random.randint(0,255,(30,112,112,3),dtype=np.uint8)

tensor = pre.preprocess(dummy)

print(tensor.reshape)