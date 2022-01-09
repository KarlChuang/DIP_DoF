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

def sharpe(dir, filename, label):
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
            imgLP = Gaussian33(imgCrop)
            imgLP = Laplacian(imgLP)
            imgLP = np.array(imgLP)
            shar = np.var(imgLP)
            if shar < 2000:
                continue
            fn = str(idx) + '_' + filename
            dd = dir.replace('image', 'imageSharp')
            imgCrop.save(path.join(dd, fn))
            label['file'].append(fn)
            label['sharp'].append(shar)
            idx += 1
        top += 32


def sharpAll(rootdir):
    label = { 'file': [], 'sharp': [] }
    labelPath = rootdir.replace('image', 'sharp.csv')
    for subdir, dirs, files in walk(rootdir):
        for file in files:
            pd.DataFrame(label).to_csv(labelPath, index=False)
            sharpe(subdir, file, label)

def select_sharp(sharp):
    rootdir = path.join('data', 'train', 'image')
    labelPath = rootdir.replace('image', 'sharp2.csv')
    df = pd.read_csv(labelPath)
    print(df.shape)
    data = df.loc[df['sharp'] > sharp].reset_index(drop=True)
    print(data.shape)
    # labelPath = rootdir.replace('image', 'sharp4.csv')
    # data.to_csv(labelPath, index=False)

    rootdir = path.join('data', 'test', 'image')
    labelPath = rootdir.replace('image', 'sharp2.csv')
    df = pd.read_csv(labelPath)
    print(df.shape)
    data = df.loc[df['sharp'] > sharp].reset_index(drop=True)
    print(data.shape)
    # labelPath = rootdir.replace('image', 'sharp4.csv')
    # data.to_csv(labelPath, index=False)

    rootdir = path.join('data', 'valid', 'image')
    labelPath = rootdir.replace('image', 'sharp2.csv')
    df = pd.read_csv(labelPath)
    print(df.shape)
    data = df.loc[df['sharp'] > sharp].reset_index(drop=True)
    print(data.shape)
    # labelPath = rootdir.replace('image', 'sharp4.csv')
    # data.to_csv(labelPath, index=False)

def move_to_valid():
    rootdir = path.join('data', 'train', 'image')
    labelPath = rootdir.replace('image', 'sharp2.csv')
    df = pd.read_csv(labelPath)
    N = int(df.shape[0] / 5 * 4)
    train_df = df.iloc[:N,:].reset_index(drop=True)
    valid_df = df.iloc[N:,:].reset_index(drop=True)
    train_df.to_csv(labelPath, index=False)
    labelPath = labelPath.replace('train', 'valid')
    valid_df.to_csv(labelPath, index=False)

    from_dir = rootdir.replace('image', 'imageSharp')
    to_dir = from_dir.replace('train', 'valid')
    for file in valid_df['file']:
        print(file)
        shutil.move(path.join(from_dir, file), path.join(to_dir, file))

def check(rootdir):
    for subdir, dirs, files in walk(rootdir):
        print(len(files))
    sharpfile = rootdir.replace('imageBlur5000', 'groundTruth5000.csv')
    df = pd.read_csv(sharpfile)
    print(df.shape)

def createLabel(rootdir):
    sharpfile = rootdir.replace('imageSharp', 'sharp2.csv')
    df = pd.read_csv(sharpfile)
    labelDf = { 'file': [], 'label': [] }
    for file in df['file']:
        blur(rootdir, file, labelDf)
    trainGT = pd.DataFrame(labelDf)
    trainGT.to_csv(rootdir.replace('imageSharp', 'groundTruth5000.csv'), index=False)


def blur(dir, filename, gt):
    img = Image.open(path.join(dir, filename)).convert('L')
    # print(path.join(dir, filename)
    imgArr = np.array(img)
    # for i in range(20):
    z = int(np.random.rand(1)[0] * 20)
    newArr = gaussian_filter(imgArr, sigma=z)
    newArr.astype(int)
    data = Image.fromarray(newArr, 'L')
    # fn = str(i) + '_' + filename
    fn = filename
    dd = dir.replace('imageSharp', 'imageBlur5000')
    gt['file'].append(fn)
    gt['label'].append(z)
    data.save(path.join(dd, fn))

if __name__ == '__main__':
    # rootdir = path.join('data', 'train', 'image')
    # sharpAll(rootdir)

    # rootdir = path.join('data', 'test', 'image')
    # sharpAll(rootdir)

    # select_sharp(5000)
    # move_to_valid()



    # createLabel(path.join('data', 'train', 'imageSharp'))
    # createLabel(path.join('data', 'valid', 'imageSharp'))
    # createLabel(path.join('data', 'test', 'imageSharp'))

    check(path.join('data', 'train', 'imageBlur5000'))
    check(path.join('data', 'valid', 'imageBlur5000'))
    check(path.join('data', 'test', 'imageBlur5000'))