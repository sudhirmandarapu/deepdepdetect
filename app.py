from preprocessor import Preprocessor
import transcript
import numpy as np
from models.log_reg import LogRegModel
from models.dnn import DNN, CrossValidationDNN, rmse
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC

# 'dev_split_Depression_AVEC2017.csv',
#    'train_split_Depression_AVEC2017.csv'

transcripts = transcript.get_transcripts_in_path('./transcripts')
p = Preprocessor('./transcripts', transcripts, 'train_split_Depression_AVEC2017.csv')
x, y, titles = p.get_all_transcript_features({
    # "liwc": True,
    # "liwc_indexes": list(range(80)),
    # "sentiment": True,
    "lda": True,
    # "antidepressants": True,
    # "absolutist": True
})

p_final = Preprocessor('./transcripts', transcripts, 'dev_split_Depression_AVEC2017.csv')
x_final, y_final, titles = p_final.get_all_transcript_features({
    # "liwc": True,
    # "liwc_indexes": list(range(80)),
    # "sentiment": True,
    "lda": True,
    # "antidepressants": True,
    # "absolutist": True
})

# Feature selection.
#sel = SelectKBest(chi2, k=90)
#print(x.shape)
# lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(x, y)
# model = SelectFromModel(lsvc, prefit=True)
# x = model.transform(x)
#x = sel.fit_transform(x, y)
# print(new_x.shape)A


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

# log_reg_model = LogRegModel(train_x, train_y, test_x, test_y)
# log_reg_model.train()

dnn = DNN(train_x, train_y, test_x, test_y, 0.001)
model = dnn.train()

print(model.metrics_names)
print(model.test_on_batch(x_final, y_final))

#rint("test accuracy:", sum(corrects)/len(corrects))
#print("rmse: ", rmse(y_final, predictions))

# cv_dnn = CrossValidationDNN(train_x, train_y, test_x, test_y, x_final, y_final, 0.001, 3)
# cv_dnn.train_with_cross_validation()

