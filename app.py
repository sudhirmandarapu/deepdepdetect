from preprocessor import Preprocessor
import transcript
import numpy as np
from models.log_reg import LogRegModel
from models.dnn import CrossDNN


transcripts = transcript.get_transcripts_in_path('./transcripts')
p = Preprocessor('./transcripts', transcripts)
x, y = p.get_all_transcript_features({
    "liwc_indexes": list(range(80)),
    "sentiment": True
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

log_reg_model = LogRegModel(train_x, train_y, test_x, test_y)
#  log_reg_model.train()

print(1-sum(test_y)/len(test_y))
dnn = CrossDNN(train_x, train_y, test_x, test_y, 0.01)
model = dnn.train()

exit()

print(train_x)

predictions = model.predict(train_x)
i = 0
while i < len(train_y):
    print(train_y[i], round(predictions[i][0]*100))
    i += 1
