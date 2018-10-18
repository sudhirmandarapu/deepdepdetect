from preprocessor import Preprocessor
import transcript
import numpy as np
from models.log_reg import LogRegModel
from models.dnn import DNN, CrossValidationDNN
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.svm import LinearSVC

# 'dev_split_Depression_AVEC2017.csv',
#    'train_split_Depression_AVEC2017.csv'

transcripts = transcript.get_transcripts_in_path('./transcripts')
p = Preprocessor('./transcripts', transcripts, 'train_split_Depression_AVEC2017.csv')
x, y = p.get_all_transcript_features({
    "liwc": True,
    "liwc_indexes": list(range(80)),
    "sentiment": True,
    "lda": True,
    "antidepressants": True,
    "absolutist": True
})

p_final = Preprocessor('./transcripts', transcripts, 'dev_split_Depression_AVEC2017.csv')
x_final, y_final = p_final.get_all_transcript_features({
    "liwc": True,
    "liwc_indexes": list(range(80)),
    "sentiment": True,
    "lda": True,
    "antidepressants": True,
    "absolutist": True
})

# Feature selection.
#sel = SelectKBest(chi2, k=90)
#print(x.shape)
# lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(x, y)
# model = SelectFromModel(lsvc, prefit=True)
# x = model.transform(x)
#x = sel.fit_transform(x, y)
# print(new_x.shape)A

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
# log_reg_model.train()

# dnn = DNN(train_x, train_y, test_x, test_y, 0.002)
# model = dnn.train()

'''
predictions = model.predict(x_final)
corrects = []
for i in range(len(predictions)):
    corrects.append(y_final[i] == round(predictions[i][0]))

print(sum(corrects)/len(corrects))
'''

cv_dnn = CrossValidationDNN(train_x, train_y, test_x, test_y, 0.001, 3)
cv_dnn.train_with_cross_validation()
