import numpy as np
import glob
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from hausdorff import hausdorff_distance
def read_nii(path):
    itk_img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itk_img)
    return img
class Score():
    def __init__(self):
        super(Score, self).__init__()
        self.dice = 0
        self.sen = 0
        self.spe = 0
        self.hua = 0
        self.i = 0
    def zero(self):
        self.dice = 0
        self.sen = 0
        self.spe = 0
        self.hua = 0
        self.i = 0


    def _Dice(self, predict, mask):
        predict = predict.flatten()
        mask = mask.flatten()
        TP = (predict * mask).sum()
        dice = 2 * TP / (predict.sum() + mask.sum())
        return dice


    def _sensitivity(self, predict, mask):
        predict = predict.flatten()
        mask = mask.flatten()
        TP = (predict * mask).sum()
        return TP / predict.sum()

    def _specificity(self, predict, mask):
        predict = predict.flatten()
        mask = mask.flatten()
        TN = np.sum(predict[mask == 0] == 0)
        FP = predict.sum() - (predict * mask).sum()
        return TN / (TN + FP)

    def _hausdorff(self, predict, mask):
        return hausdorff_distance(predict, mask, distance="manhattan")


    def DSSH_scores(self, predict, mask):
        self.dice = self.dice + self._Dice(predict, mask)
        self.sen = self.sen + self._sensitivity(predict, mask)
        self.spe = self.spe + self._specificity(predict, mask)
        self.hua = self.hua + self._hausdorff(predict, mask)
        self.i = self.i + 1

if __name__ == "__main__":
    # F:\Project\make_dataset\PLOP\save\*.nii.gz
    path1 = r"F:\Project\make_dataset\PLOP\Test\*\*_seg.nii.gz"
    path2 = r"F:\Project\make_dataset\PLOP\train_save\*.npy"
    pre = glob.glob(path1)
    lbl = glob.glob(path2)
    file = "./train_test.xlsx"
    result = {"ET_Dice": [], "ET_sen": [], "ET_spe": [], "ET_hua": [],
              "WT_Dice": [], "WT_sen": [], "WT_spe": [], "WT_hua": [],
              "TC_Dice": [], "TC_sen": [], "TC_spe": [], "TC_hua": []}

    ET_source = Score()
    WT_source = Score()
    TC_source = Score()
    for j in range(len(pre)):
        pred = read_nii(pre[j]).astype(np.uint8)
        label = read_nii(lbl[j]).astype(np.uint8)
        for i in range(155):
            if 1 in np.unique(label[i]) and 1 in np.unique(pred[i]):
                predict = np.zeros((pred[i].shape[0], pred[i].shape[1]))
                predict[pred[i] == 1] = 1
                mask = np.zeros((label[i].shape[0], label[i].shape[1]))
                mask[label[i] == 1] = 1
                ET_source.DSSH_scores(predict, mask)
            if 2 in np.unique(label[i]) and 2 in np.unique(pred[i]):
                predict = np.zeros((pred[i].shape[0], pred[i].shape[1]))
                predict[pred[i] == 2] = 1
                mask = np.zeros((label[i].shape[0], label[i].shape[1]))
                mask[label[i] == 2] = 1
                WT_source.DSSH_scores(predict, mask)
            if 4 in np.unique(label[i]) and 4 in np.unique(pred[i]):
                predict = np.zeros((pred[i].shape[0], pred[i].shape[1]))
                predict[pred[i] == 4] = 1
                mask = np.zeros((label[i].shape[0], label[i].shape[1]))
                mask[label[i] == 4] = 1
                TC_source.DSSH_scores(predict, mask)
        if ET_source.i != 0:
            result["ET_Dice"].append(ET_source.dice / ET_source.i)
            result["ET_sen"].append(ET_source.sen / ET_source.i)
            result["ET_spe"].append(ET_source.spe / ET_source.i)
            result["ET_hua"].append(ET_source.hua / ET_source.i)
        else:
            result["ET_Dice"].append(0)
            result["ET_sen"].append(0)
            result["ET_spe"].append(0)
            result["ET_hua"].append(0)
        if WT_source.i != 0:
            result["WT_Dice"].append(WT_source.dice / WT_source.i)
            result["WT_sen"].append(WT_source.sen / WT_source.i)
            result["WT_spe"].append(WT_source.spe / WT_source.i)
            result["WT_hua"].append(WT_source.hua / WT_source.i)
        else:
            result["WT_Dice"].append(0)
            result["WT_sen"].append(0)
            result["WT_spe"].append(0)
            result["WT_hua"].append(0)
        if TC_source.i != 0:
            result["TC_Dice"].append(TC_source.dice / TC_source.i)
            result["TC_sen"].append(TC_source.sen / TC_source.i)
            result["TC_spe"].append(TC_source.spe / TC_source.i)
            result["TC_hua"].append(TC_source.hua / TC_source.i)
        else:
            result["TC_Dice"].append(0)
            result["TC_sen"].append(0)
            result["TC_spe"].append(0)
            result["TC_hua"].append(0)

        ET_source.zero()
        WT_source.zero()
        TC_source.zero()
    df1 = pd.DataFrame(result)
    df1.to_excel(file)







