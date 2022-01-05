import numpy as np
from PIL import Image, ImageFilter
from os import path, walk
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import shutil

def Gaussian33(img):
    kernel = ImageFilter.Kernel((3, 3), (1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16), 1, 0)
    im2 = img.filter(kernel)
    return im2

def Laplacian(img):
    kernel = ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0)
    im2 = img.filter(kernel)
    return im2

def sharpe(dir, filename):
    img = Image.open(path.join(dir, filename)).convert('L')
    print(path.join(dir, filename))
    # crop
    (height, width) = img.size
    (top, idx) = (0, 1)
    while top + 32 <= height:
        left = 0
        while left + 32 <= width:
            imgCrop = img.crop((top, left, top + 32, left + 32))
            left += 32
            imgCrop = Gaussian33(imgCrop)
            imgLP = Laplacian(imgCrop)
            imgLP = np.array(imgLP)
            if np.var(imgLP) < 1000:
                continue
            fn = str(idx) + '_' + filename
            dd = dir.replace('image', 'imageSharp')
            imgCrop.save(path.join(dd, fn))
            idx += 1
        top += 32

# def blur(dir, filename)
#     img = Image.open(path.join(dir, filename)).convert('L')
#     print(path.join(dir, filename))
#     for z in range(50):
#         newArr = [gaussian_filter(imgArrC, sigma=z * 0.2) for imgArrC in imgArr]
#         newArr = np.array(newArr)
#         newArr = np.transpose(newArr, (1, 2, 0))
#         newArr.astype(int)
#         data = Image.fromarray(newArr, 'RGB')
#         fn = str(z) + '_' + filename
#         fn = fn.replace('jpg', 'png') 
#         dd = dir.replace('image', 'imageBlur')
#         gt['file'].append(fn)
#         gt['label'].append(z)
#         data.save(path.join(dd, fn))


if __name__ == '__main__':
    # rootdir = path.join('data', 'train', 'image')
    # for subdir, dirs, files in walk(rootdir):
    #     for file in files:
    #         sharpe(subdir, file)

    # rootdir = path.join('data', 'val', 'image')
    # for subdir, dirs, files in walk(rootdir):
    #     for file in files:
    #         sharpe(subdir, file)

    trainGT = { 'file': [], 'label': [] }
    rootdir = path.join('data', 'train', 'imageSharp')
    for subdir, dirs, files in walk(rootdir):
        print(len(files))
        # for file in files:
        #     sharpe(subdir, file)
    # trainGT = pd.DataFrame(trainGT)
    # trainGT.to_csv(rootdir.replace('imageSharp', 'groundTruth.csv'))

    testGT = { 'file': [], 'label': [] }
    rootdir = path.join('data', 'valid', 'imageSharp')
    for subdir, dirs, files in walk(rootdir):
        print(len(files))
    #     for file in files:
    #         sharpe(subdir, file)
    # testGT = pd.DataFrame(testGT)
    # testGT.to_csv(rootdir.replace('imageSharp', 'groundTruth.csv'))

    testGT = { 'file': [], 'label': [] }
    rootdir = path.join('data', 'test', 'imageSharp')
    for subdir, dirs, files in walk(rootdir):
        print(len(files))
    #     for file in files:
    #         sharpe(subdir, file)
    # testGT = pd.DataFrame(testGT)
    # testGT.to_csv(rootdir.replace('imageSharp', 'groundTruth.csv'))