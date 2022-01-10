from PIL import Image, ImageFilter
import numpy as np


# def Laplacian(img):
#     kernel = ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 0.001,
#                                 0)
#     im2 = img.filter(kernel)
#     return im2


def combine(name):
    imageF = Image.open('s_001-01.jpg')
    imageFd = Image.open('s_001-01_depth.png')
    imageB = Image.open('s_001-02.jpg')
    imageBd = Image.open('s_001-02_depth.png')
    # im2 = Laplacian(imageFd)
    # im2.show()
    # exit()

    imgArrF = np.array(imageF)
    imgArrFd = np.array(imageFd)
    imgArrB = np.array(imageB)
    imgArrBd = np.array(imageBd)

    imgArrN = np.copy(imgArrF)
    imgArrN[imgArrFd > imgArrBd, :] = imgArrB[imgArrFd > imgArrBd, :]
    data = Image.fromarray(imgArrN.astype('uint8'), 'RGB')
    data.show()
    exit()

if __name__ == '__main__':
    combine('ttt.png')
