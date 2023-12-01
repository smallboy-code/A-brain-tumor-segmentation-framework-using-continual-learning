import os
import glob
import random
import numpy as np
import SimpleITK as sitk
from Tool import crop_ceter, make_file, normalize, GUI_0_255
"""
VOCdevkit/VOC2012/JPEGImages
SegmentationClassAug
"""
root = "E:\BraTS2020_TrainingData"
# train_name = os.listdir(os.path.join(root, "MICCAI_BraTS2020_TrainingData"))
# # test_name = os.listdir(os.path.join(root, "BraTS2020_test"))
# print("train file number: {}".format(len(train_name)))
# # print("test file number: {}".format(len(test_name)))
#
#
# """
# train_data slice
# """
# flair_name = "_flair.nii.gz"
# t1_name = "_t1.nii.gz"
# t1ce_name = "_t1ce.nii.gz"
# t2_name = "_t2.nii.gz"
# mask_name = "_seg.nii.gz"
#
# bratshgg_path = os.path.join(root, "MICCAI_BraTS2020_TrainingData")
# outputImg_path = os.path.join(root, "PLOP_train_image")
# outputMask_path = os.path.join(root, "PLOP_train_label")
#
# make_file(outputImg_path)
# make_file(outputMask_path)
# ridm = (160, 160)
# for i in range(len(train_name)):
#     brats_subset_path = bratshgg_path + "/" + train_name[i] + "/"
#     # 获取每个病例的四个模态及Mask的路径
#     flair_image = brats_subset_path + train_name[i] + flair_name
#     t1_image = brats_subset_path + train_name[i] + t1_name
#     t1ce_image = brats_subset_path + train_name[i] + t1ce_name
#     t2_image = brats_subset_path + train_name[i] + t2_name
#     mask_image = brats_subset_path + train_name[i] + mask_name
#     # 获取每个病例的四个模态及Mask数据
#     flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
#     t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
#     t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
#     t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
#     mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
#     # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
#     flair_array = sitk.GetArrayFromImage(flair_src)
#     t1_array = sitk.GetArrayFromImage(t1_src)
#     t1ce_array = sitk.GetArrayFromImage(t1ce_src)
#     t2_array = sitk.GetArrayFromImage(t2_src)
#     mask_array = sitk.GetArrayFromImage(mask)
#     # 插入人工切片
#     flair_new = np.zeros((160, 240, 240)).astype(np.float64)
#     flair_new[3:158, :, :] = flair_array
#     t1_new = np.zeros((160, 240, 240)).astype(np.float64)
#     t1_new[3:158, :, :] = t1_array
#     t1ce_new = np.zeros((160, 240, 240)).astype(np.float64)
#     t1ce_new[3:158, :, :] = t1ce_array
#     t2_new = np.zeros((160, 240, 240)).astype(np.float64)
#     t2_new[3:158, :, :] = t2_array
#     mask_new = np.zeros((160, 240, 240)).astype(np.uint8)
#     mask_new[3:158, :, :] = mask_array
#
#     # 裁剪(偶数才行)
#     flair_crop = crop_ceter(flair_new, 160, 160)  # 160*160*160
#     t1_crop = crop_ceter(t1_new, 160, 160)
#     t1ce_crop = crop_ceter(t1ce_new, 160, 160)
#     t2_crop = crop_ceter(t2_new, 160, 160)
#     mask_crop = crop_ceter(mask_new, 160, 160)
#
#     flair_crop = normalize(flair_crop)
#     t1_crop = normalize(t1_crop)
#     t1ce_crop = normalize(t1ce_crop)
#     t2_crop = normalize(t2_crop)
#
#     flair_crop = GUI_0_255(flair_crop)
#     t1_crop = GUI_0_255(t1_crop)
#     t1ce_crop = GUI_0_255(t1ce_crop)
#     t2_crop = GUI_0_255(t2_crop)
#
#     print("train data: {}".format(train_name[i]))
#     mask_crop[mask_crop == 4] = 3
#     # 切片处理,并去掉没有病灶的切片
#     for n_slice in range(flair_crop.shape[0]):
#         if np.sum(mask_crop[n_slice, :, :] != 0) > 10:
#             maskImg = mask_crop[n_slice, :, :]
#             maskImg = maskImg.astype(np.uint8)
#
#             FourModelImageArray = np.zeros((160, 160, 4), np.float64)
#
#             flairImg = flair_crop[n_slice, :, :]
#             flairImg = flairImg.astype(np.float64)
#             FourModelImageArray[:, :, 0] = flairImg
#
#             t1Img = t1_crop[n_slice, :, :]
#             t1Img = t1Img.astype(np.float64)
#             FourModelImageArray[:, :, 1] = t1Img
#
#             t1ceImg = t1ce_crop[n_slice, :, :]
#             t1ceImg = t1ceImg.astype(np.float64)
#             FourModelImageArray[:, :, 2] = t1ceImg
#
#             t2Img = t2_crop[n_slice, :, :]
#             t2Img = t2Img.astype(np.float64)
#             FourModelImageArray[:, :, 3] = t2Img
#
#             imagepath = outputImg_path + "/" + train_name[i] + "_" + str(n_slice) + ".npy"
#             maskpath = outputMask_path + "/" + train_name[i] + "_" + str(n_slice) + ".npy"
#
#             np.save(imagepath, FourModelImageArray)  # (320,320,4) np.float dtype('float64')
#             np.save(maskpath, maskImg)  # (320, 320) dtype('uint8') 值为0 1 2 4
# print("Done！")
#

"""
test data slice
"""
# bratshgg_path = os.path.join(root, "BraTS2020_test")
# outputImg_path = os.path.join(root, "test_image")
# make_file(outputImg_path)
#
# for i in range(len(test_name)):
#     brats_subset_path = bratshgg_path + "/" + test_name[i] + "/"
#     # 获取每个病例的四个模态及Mask的路径
#     flair_image = brats_subset_path + test_name[i] + flair_name
#     t1_image = brats_subset_path + test_name[i] + t1_name
#     t1ce_image = brats_subset_path + test_name[i] + t1ce_name
#     t2_image = brats_subset_path + test_name[i] + t2_name
#
#     # 获取每个病例的四个模态及Mask数据
#     flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
#     t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
#     t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
#     t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
#
#     # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
#     flair_array = sitk.GetArrayFromImage(flair_src)
#     t1_array = sitk.GetArrayFromImage(t1_src)
#     t1ce_array = sitk.GetArrayFromImage(t1ce_src)
#     t2_array = sitk.GetArrayFromImage(t2_src)
#
#     # 插入人工切片(前3, 后2)
#     flair_new = np.zeros((160, 240, 240)).astype(np.float64)
#     flair_new[3:158, :, :] = flair_array
#     t1_new = np.zeros((160, 240, 240)).astype(np.float64)
#     t1_new[3:158, :, :] = t1_array
#     t1ce_new = np.zeros((160, 240, 240)).astype(np.float64)
#     t1ce_new[3:158, :, :] = t1ce_array
#     t2_new = np.zeros((160, 240, 240)).astype(np.float64)
#     t2_new[3:158, :, :] = t2_array
#
#     # crop
#     flair_crop = crop_ceter(flair_new, 160, 160)  # 160*160*160
#     t1_crop = crop_ceter(t1_new, 160, 160)
#     t1ce_crop = crop_ceter(t1ce_new, 160, 160)
#     t2_crop = crop_ceter(t2_new, 160, 160)
#
#     flair_crop = normalize(flair_crop)
#     t1_crop = normalize(t1_crop)
#     t1ce_crop = normalize(t1ce_crop)
#     t2_crop = normalize(t2_crop)
#
#     flair_crop = GUI_0_255(flair_crop)
#     t1_crop = GUI_0_255(t1_crop)
#     t1ce_crop = GUI_0_255(t1ce_crop)
#     t2_crop = GUI_0_255(t2_crop)
#
#     print("test data: {}".format(test_name[i]))
#     # 切片处理,并去掉没有病灶的切片
#     data = np.zeros((160, 160, 160, 4))
#     for n_slice in range(flair_crop.shape[0]):
#         FourModelImageArray = np.zeros((160, 160, 4), np.float64)
#
#         flairImg = flair_crop[n_slice, :, :]
#         flairImg = flairImg.astype(np.float64)
#         FourModelImageArray[:, :, 0] = flairImg
#
#         t1Img = t1_crop[n_slice, :, :]
#         t1Img = t1Img.astype(np.float64)
#         FourModelImageArray[:, :, 1] = t1Img
#
#         t1ceImg = t1ce_crop[n_slice, :, :]
#         t1ceImg = t1ceImg.astype(np.float64)
#         FourModelImageArray[:, :, 2] = t1ceImg
#
#         t2Img = t2_crop[n_slice, :, :]
#         t2Img = t2Img.astype(np.float64)
#         FourModelImageArray[:, :, 3] = t2Img
#
#         imagepath = outputImg_path + "/" + test_name[i] + "/" + str(n_slice) + ".npy"
#
#         data[n_slice, :, :, :] = FourModelImageArray
#     imagepath = outputImg_path + "/" + test_name[i] + ".npy"
#     np.save(imagepath, data)
# print("Done！")


"""
make index list
"""
import os
import glob
import random
def read_npy(path):
    img_name = glob.glob(path)
    train_aug = []
    for i in range(len(img_name)):
        name = img_name[i].split("\\")[-1]
        t = "\PLOP_train_image\\" + name + " " + "\PLOP_train_label\\" + name
        train_aug.append(t)
    random.shuffle(train_aug)
    return train_aug

def make_txt(train_path, val_path, train_aug):
    f1 = open(train_path, 'w')
    f2 = open(val_path, 'w')
    listen = int(0.85 * len(train_aug))
    for i in range(len(train_aug)):
        if i < listen:
            f2.write(train_aug[i] + '\n')
        else:
            f1.write(train_aug[i] + '\n')
    f1.close()
    f2.close()


def make_file_path(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def Make(image_path, txt_path):
    make_file_path(txt_path)
    train_aug = read_npy(image_path)
    train_path = os.path.join(txt_path, "val.txt")
    train_aug_path = os.path.join(txt_path, "train_aug.txt")
    make_txt(train_path, train_aug_path, train_aug)

image_path = os.path.join(root, "PLOP_train_image", "*.npy")
txt_path = root
Make(image_path, txt_path)

