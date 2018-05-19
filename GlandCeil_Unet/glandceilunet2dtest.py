from __future__ import division
from unet2d.model_GlandCeil import unet2dModule
import numpy as np
import pandas as pd
import cv2


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('D:\Project\python\GlandCeil_Unet/GlandsMask.csv')
    csvimagedata = pd.read_csv('D:\Project\python\GlandCeil_Unet/GlandsImage.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = unet2dModule(512, 512, channels=3, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "D:\Project\python\GlandCeil_Unet\model\\unet2dglandceil.pd",
                 "D:\Project\python\GlandCeil_Unet\log", 0.0005, 0.8, 100000, 2)


def predict():
    true_img = cv2.imread("D:\Data\GlandCeil\Test\Image\\testA_55.bmp", cv2.IMREAD_COLOR)
    test_images = true_img.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    unet2d = unet2dModule(512, 512, 3)
    predictvalue = unet2d.prediction("D:\Project\python\GlandCeil_Unet\model\\unet2dglandceil.pd",
                                     test_images)
    cv2.imwrite("mask1.bmp", predictvalue)


def main(argv):
    if argv == 1:
        train()
    if argv == 2:
        predict()


if __name__ == "__main__":
    main(2)
