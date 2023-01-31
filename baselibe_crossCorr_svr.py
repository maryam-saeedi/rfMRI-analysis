from dataGenerator import DataGenerator
from sklearn import svm
import pandas as pd
import numpy as np  

dataGen = DataGenerator()
validRegions = dataGen.getValidRegions()

features_per_sbj = {}

for sbj in dataGen.subjects:
    data = []
    dataFrame = pd.read_csv("../Data/"+sbj+"/Copy of fMRI_aparc.a2009s+aseg_mean_ROI_timecourse.csv", header=None, index_col=0)
    dataFrame.replace(r'\\n','', regex=True, inplace=True)
    dataFrame.drop(columns=[250], inplace=True)
    for region in validRegions:
        data.append(dataFrame.loc[region].values)
    data = np.array(data, float)
    corr = np.corrcoef(data)
    flatten = []
    for i in range(1, corr.shape[0]):
        for j in range(i):
            flatten.append(corr[i,j])
    features_per_sbj[sbj] = flatten

subjects = dataGen.subjects
np.random.shuffle(subjects)

regr = svm.SVR()
mse = 0
for test_sbj_idx in range(len(subjects)):
    features = []
    labels = []
    for idx in range(len(subjects)):
        if test_sbj_idx == idx:
            continue
        data = features_per_sbj[subjects[idx]]
        label = dataGen.getFOG_QValue(subjects[idx])
        features.append(data)
        labels.append(label)
    features = np.array(features)
    regr.fit(features, labels)

    test_sbj = subjects[test_sbj_idx]
    test = features_per_sbj[test_sbj]
    label = dataGen.getFOG_QValue(test_sbj)
    
    predicted = regr.predict([test])[0]
    mse += (predicted-label)**2

mse/=len(subjects)
print("MSE is: ", mse)
