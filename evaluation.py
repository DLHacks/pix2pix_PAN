import argparse
import numpy as np
from PIL import Image
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', required=True)
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, args.imgs_path)
    print(path)
    mean_MSE = 0
    count = 0
    for name in os.listdir(path):
        if name[-10:] == "fake_B.png":
            fake_path = path + name
            real_path = path + name[:-10] + "real_B.png"
            fake_img = np.array(Image.open(fake_path))
            real_img = np.array(Image.open(real_path))
            MSE = np.sum((real_img - fake_img)**2) / real_img.shape[0] / real_img.shape[1] / real_img.shape[2]
            mean_MSE += MSE
            count += 1
    mean_MSE = mean_MSE / count
    PSNR = 10 * np.log10(255*255/mean_MSE)
    print(PSNR)
