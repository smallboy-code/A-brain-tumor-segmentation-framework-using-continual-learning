import os
import cv2
import glob
import numpy as np
from PIL import Image
import SimpleITK as sitk
from torchvision import transforms

val_train = transforms.Compose(
    [
        transforms.Resize(size=(160, 160))
    ]
)

def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_nii_gz(path):
    image = sitk.ReadImage(path, sitk.sitkInt16)
    image = sitk.GetArrayFromImage(image)
    return image

def make_nii_gz(data, mask, path):
    data = sitk.GetImageFromArray(data)
    origin_img = mask.GetOrigin()
    spacing = mask.GetSpacing()
    direction = mask.GetDirection()

    data.SetOrigin(origin_img)
    data.SetSpacing(spacing)
    data.SetDirection(direction)
    sitk.WriteImage(data, path)

save_path1 = "./test_result"
#save_path2 = "./val_result"
make_file(save_path1)

path1 = r"./test_save/*.npy"
#path2 = r"./val_save/*.npy"
name_path = glob.glob(path1)

mask_path = r"./brats_train/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz"
mask = sitk.ReadImage(mask_path)
for i in range(len(name_path)):
    data = np.zeros((160, 160, 160)).astype(np.uint8)
    a = name_path[i].split("/")[2].split(".")[0]

    # 都是[160, 640， 640]
    image1 = np.load(name_path[i]).astype(np.uint8)
    for j in range(160):
        #img1 = cv2.resize(image1[j, :, :], (160, 160), interpolation=cv2.INTER_NEAREST)
        #data[j, :, :] = img1
        data[j, :, :] = image1[j, :, :]
    save = np.zeros((160, 240, 240)).astype(np.int16)
    save[:, 40:200, 40:200] = data
    BraTS = save[3:158, :, :]
    make_nii_gz(BraTS, mask, os.path.join(save_path1, a + ".nii.gz"))







