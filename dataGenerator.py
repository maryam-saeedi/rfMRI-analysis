import pandas as pd

class DataGenerator:
    def __init__(self):
        self.metadata = pd.read_csv("../Data/sort_FOG_MCI.csv", index_col="PCSID")
        self.subjects = self.metadata.index.values
        self.__setRegions("Copy of fMRI_aparc.a2009s+aseg_mean_ROI_timecourse")
        self.__setFOG_QValue()

    def __setFOG_QValue(self):
        self.FOG_per_sbj = {}
        for sbj in self.subjects:
            self.FOG_per_sbj[sbj] = self.metadata.loc[sbj]["FOG-Q Total"]

    def __setRegions(self, file):
        self.fileNameTemplate = "Copy of fMRI_aparc.a2009s+aseg_mean_ROI_timecourse"
        sampleFile = pd.read_csv("../Data/UOA0040/"+self.fileNameTemplate+".csv", header=None, index_col=0)
        self.regions = sampleFile.index.values


    def getFOG_QValue(self, subject):
        return self.FOG_per_sbj[subject]
        
    def getBinaryData(self, threshold):
        pos_sbj = [sbj for sbj, FOG_Q in self.FOG_per_sbj.items() if FOG_Q > threshold]
        neg_sbj = [sbj for sbj, FOG_Q in self.FOG_per_sbj.items() if FOG_Q < threshold]
        return pos_sbj, neg_sbj

    def getValidRegions(self) -> list:
        ROI_sbj = {}
        
        for region in self.regions:
            ROI_sbj[region] = []

        for s in self.subjects:
            dataFilePath = "../Data/"+s+"/"+self.fileNameTemplate+".csv"
            sbj = dataFilePath.replace('\\','/').split('/')[-2]
            dataFile = pd.read_csv(dataFilePath, header=None, index_col=0)
            dataFile = dataFile.replace(r'\\n','', regex=True)
            dataFile = dataFile.drop(columns=[250])
            for region in dataFile.index.values:
                if dataFile.loc[region].all():
                    ROI_sbj[region].append(sbj)

        validROIs = []
        for region, subjetcs in ROI_sbj.items():    
            if len(subjetcs) == len(self.subjects):
                validROIs.append(region)

        return validROIs
