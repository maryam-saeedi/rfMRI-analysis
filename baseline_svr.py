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
    features_per_sbj[sbj] = data

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
        for timestamp in range(data.shape[1]):
            features.append(data[:,timestamp])
            labels.append(label)

    features = np.array(features)
    regr.fit(features, labels)

    test_sbj = subjects[test_sbj_idx]
    data = features_per_sbj[test_sbj]
    label = dataGen.getFOG_QValue(test_sbj)
    local_mse = 0
    for timestamp in range(data.shape[1]):
        test = [data[:,timestamp]]
        predicted = regr.predict(test)[0]
        local_mse += (predicted-label)**2
    local_mse/=data.shape[1]
    mse+=local_mse
mse/=len(subjects)
print("MSE is: ", mse)
