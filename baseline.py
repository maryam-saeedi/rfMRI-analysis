from dataGenerator import DataGenerator
from sklearn import svm
import pandas as pd
import numpy as np  

dataGen = DataGenerator()
posSbj, negSbj = dataGen.getBinaryData(7.5)
validRegions = dataGen.getValidRegions()

features_per_sbj = {}

for sbj in posSbj+negSbj:
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

clf = svm.SVC()
acc = 0
for test_sbj_idx in range(len(subjects)):
    features = []
    labels = []
    for idx in range(len(subjects)):
        if test_sbj_idx == idx:
            continue
        data = features_per_sbj[subjects[idx]]
        if subjects[idx] in posSbj:
            label=1
        else:
            label = 0
        for timestamp in range(data.shape[1]):
            features.append(data[:,timestamp])
            labels.append(label)

    features = np.array(features)
    clf.fit(features, labels)

    test_sbj = subjects[test_sbj_idx]
    data = features_per_sbj[test_sbj]
    if test_sbj in posSbj:
        label = 1
    else:
        label = 0
    local_acc = 0
    for timestamp in range(data.shape[1]):
        test = [data[:,timestamp]]
        predicted = clf.predict(test)[0]
        if predicted==label:
            local_acc+=1
    local_acc/=data.shape[1]
    acc+=local_acc
acc/=len(subjects)
print("accuracy is: ", acc)
