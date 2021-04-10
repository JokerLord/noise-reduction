import argparse
import imageio
import numpy as np


def non_neg_float(value):
    ivalue = float(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError('%s is an invalid positive float\
                                         value' % value)
    return ivalue


parser = argparse.ArgumentParser()
parser.add_argument('sigma', type=non_neg_float)
parser.add_argument('input_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()

img = imageio.imread(args.input_file)
img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
noise = np.random.normal(0.0, args.sigma, img.shape)
img = img + noise
num_of_rows = img.shape[0]
num_of_cols = img.shape[1]
for i in range(num_of_rows):
    for j in range(num_of_cols):
        if img[i, j] < 0:
            img[i, j] = 0
        elif img[i, j] > 255:
            img[i, j] = 255
img = img.astype(np.uint8)
imageio.imwrite(args.output_file, img)
