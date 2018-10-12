from preprocessor import Preprocessor
import transcript
import numpy as np
from models.log_reg import LogRegModel
from models.dnn import DNN, CrossValidationDNN


transcripts = transcript.get_transcripts_in_path('./transcripts')
p = Preprocessor('./transcripts', transcripts)
x, y = p.get_all_transcript_features({
    "liwc": True,
    "liwc_indexes": list(range(80)),
    "sentiment": True,
    "lda": True
})

# up sampling
i_nd = np.where(y == 0)[0]
i_d = np.where(y == 1)[0]
i_d_upsampled = np.random.choice(i_d, size=len(i_nd), replace=True)
y = np.concatenate((y[i_d_upsampled], y[i_nd]))
x = np.concatenate((x[i_d_upsampled], x[i_nd]))

seed = 5
np.random.seed(seed)

train_index = np.random.choice(len(x), round(len(x) * 0.8), replace=False)
test_index = np.array(list(set(range(len(x))) - set(train_index)))

train_x = x[train_index]
train_y = y[train_index]
test_x = x[test_index]
test_y = y[test_index]

# log_reg_model = LogRegModel(train_x, train_y, test_x, test_y)
#  log_reg_model.train()

dnn = DNN(train_x, train_y, test_x, test_y, 0.005)
model = dnn.train()

# cv_dnn = CrossValidationDNN(train_x, train_y, test_x, test_y, 0.005, 5)
# cv_dnn.train_with_cross_validation()
