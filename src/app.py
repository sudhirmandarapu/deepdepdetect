import os
from preprocessor import Preprocessor
import transcript
import numpy as np
from models.log_reg import LogRegModel
from models.dnn import DNN, CrossValidationDNN, rmse
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC


transcripts = transcript.get_transcripts_in_path('./transcripts')
p = Preprocessor('./transcripts', transcripts, os.getenv('TRAINING_DATA_FILE'))
x, y, titles = p.get_all_transcript_features({
    "liwc": True,
    "liwc_indexes": list(range(80)),
    "sentiment": True,
    "lda": True,
    "antidepressants": True,
    "absolutist": True
})

p_final = Preprocessor('./transcripts', transcripts, os.getenv('DEV_DATA_FILE'))
x_final, y_final, titles = p_final.get_all_transcript_features({
    "liwc": True,
    "liwc_indexes": list(range(80)),
    "sentiment": True,
    "lda": True,
    "antidepressants": True,
    "absolutist": True
})

# Feature selection.
# sel = SelectKBest(chi2, k=90)
# lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(x, y)
# model = SelectFromModel(lsvc, prefit=True)
# x = model.transform(x)
# x = sel.fit_transform(x, y)


seed = 5
np.random.seed(seed)

train_index = np.random.choice(len(x), round(len(x) * 0.8), replace=False)
test_index = np.array(list(set(range(len(x))) - set(train_index)))

train_x = x[train_index]
train_y = y[train_index]
test_x = x[test_index]
test_y = y[test_index]

# up sampling
i_nd = np.where(train_y == 0)[0]
i_d = np.where(train_y == 1)[0]
i_d_upsampled = np.random.choice(i_d, size=len(i_nd), replace=True)
train_y = np.concatenate((train_y[i_d_upsampled], train_y[i_nd]))
train_x = np.concatenate((train_x[i_d_upsampled], train_x[i_nd]))

model_to_use = os.getenv('MODEL')
if model_to_use == 'LOGREG':
    log_reg_model = LogRegModel(train_x, train_y, test_x, test_y)
    model = log_reg_model.train()
elif model_to_use == os.getenv('DNN'):
    dnn = DNN(train_x, train_y, test_x, test_y, 0.001)
    model = dnn.train()
else:
    cv_dnn = CrossValidationDNN(train_x, train_y, test_x, test_y, x_final, y_final, 0.001, 3)
    model = cv_dnn.train_with_cross_validation()

print(model.metrics_names)
print(model.test_on_batch(x_final, y_final))
