import os
from sesfnet import SESF_Fuse
from tqdm import tqdm
import cv2 as cv
import warnings
import matplotlib.pyplot as plt
import sys
import depth

warnings.filterwarnings("ignore")

def run_main(input_dir, out_dir):
    sesf = SESF_Fuse('cse')
    img_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    print("original image is like {}".format(img_name[0]))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Fushing start....")
    for img in tqdm(img_name):
        print("Fusing {}".format(img))
        img1 = cv.imread(os.path.join(input_dir, img+"_1.png"))
        img2 = cv.imread(os.path.join(input_dir, img+"_2.png"))
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        fused = sesf.fuse(img1, img2)
        cv.imwrite(os.path.join(out_dir, img+".png"), cv.cvtColor(fused, cv.COLOR_RGB2BGR))
    i1 = cv.imread("data/color_lytro_01_1.png")
    i2 = cv.imread("data/color_lytro_01_2.png")
    res = cv.imread("result/color_lytro_01.png")
    i1, i2, res = cv.cvtColor(i1, cv.COLOR_BGR2RGB), cv.cvtColor(i2, cv.COLOR_BGR2RGB), cv.cvtColor(res, cv.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(i1)
    ax[1].imshow(i2)
    ax[2].imshow(res)
    
    plt.show()

if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), "data")
    output_dir = os.path.join(os.getcwd(), "result")
    run_main(input_dir,output_dir)
    depth.main("data/color_lytro_01_1.png",4,32)