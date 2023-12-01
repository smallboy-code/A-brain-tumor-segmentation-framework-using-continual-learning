import os
import numpy as np

def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

def normalize(slice, bottom=99, down=1):
    #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)
    #除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp


def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]

def GUI_0_255(data):
    ymax = 255  # 要归一的范围的最大值
    ymin = 0  # 要归一的范围的最小值
    xmax = np.max(data)
    xmin = np.min(data[data != -9])
    new_data = (ymax-ymin)*(data-xmin)/(xmax-xmin)+ymin
    new_data[new_data < 0] = 0
    new_data = new_data.astype(np.float64)
    return new_data

def GUI_0_1(data):
    ymin = np.min(data[data != -9])
    ymax = np.max(data)
    new_data = (data - ymin) / (ymax - ymin)
    new_data[new_data < 0] = -10
    return new_data

