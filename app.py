from preprocessor import Preprocessor, SplitPreprocessor
import numpy as np
from models.log_reg import LogRegModel
from models.dnn import DNN
from models.low_dnn import LowDNN
import sys

preprocessor = Preprocessor('./transcripts')

splitPreprocessor = SplitPreprocessor('./output/split_transcripts')


x, y = splitPreprocessor.get_all_transcript_features()

seed = 5
np.random.seed(seed)

train_index = np.random.choice(len(x), round(len(x) * 0.8), replace=False)
test_index = np.array(list(set(range(len(x))) - set(train_index)))

train_x = x[train_index]
train_y = y[train_index]
test_x = x[test_index]
test_y = y[test_index]

#print(train_x.shape)
#print(test_x.shape)

for i in train_x[0]: print(float(i))
#dnn = DNN(train_x, train_y, test_x, test_y, 0.01)
#model = dnn.train()

'''
thing = sum(y)/len(y)
thing_1 = 1 - thing
print(thing)
print(thing_1)
'''

#print(test_y)
#for i in model.predict(test_x): print(round(float(i)))

'''
if __name__ == '__main__':
    m_type = sys.argv[1]
    if m_type == 'log_reg':
        log_reg_model = LogRegModel(train_x, train_y, test_x, test_y)
        log_reg_model.train()
    elif m_type == 'dnn':
        dnn = DNN(train_x, train_y, test_x, test_y, 0.001)
        dnn.train()
    elif m_type == 'low':
        low_dnn = LowDNN(train_x, train_y, test_x, test_y)
        low_dnn.train()
'''