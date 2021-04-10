import argparse
import imageio
import numpy as np
import os

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def make_gray_scale(img):
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def mse(img1, img2):
    if img1.shape == img2.shape:
        return (np.sum((img1 - img2) ** 2)
                / (img1.shape[0] * img1.shape[1]))
    else:
        return None


def median(img, r):
    res = img.copy()
    tmp = np.concatenate((np.repeat(img[0].reshape(1, -1), r, axis=0), img),
                         axis=0)
    tmp = np.concatenate((tmp, np.repeat(tmp[-1].reshape(1, -1), r, axis=0)),
                         axis=0)
    tmp = np.concatenate((tmp, np.repeat(tmp[:, -1].reshape(-1, 1), r,
                                         axis=1)), axis=1)
    tmp = np.concatenate((np.repeat(tmp[:, 0].reshape(-1, 1), r, axis=1),
                          tmp), axis=1)
    num_of_rows, num_of_cols = img.shape
    for i in range(r, r + num_of_rows):
        for j in range(r, r + num_of_cols):
            res[i - r, j - r] = np.median(tmp[(i - r):(i + r + 1), (j - r):(j + r + 1)])
    return res


def calc_noise(img):
    noise = img - gaussian_filter(img, 1, mode='nearest')
    noise_level = np.std(noise)
    sigma = query(noise_level)
    noise = img - gaussian_filter(img, sigma=sigma, mode='nearest')
    return np.std(noise)


def query(noise_level):
    noises = np.logspace(0, 8, num=9, base=2.0).astype(int)
    params = np.zeros(noises.shape[0])
    for num in range(noises.shape[0]):
        sigmas = np.arange(1, 2.1, 0.001)
        mses = np.zeros(sigmas.shape)
        for i in range(sigmas.shape[0]):
            for filename in os.listdir('noise' + str(noises[num])):
                noisy_img = imageio.imread('noise' + str(noises[num]) + '\\' + filename)
                clear_img = imageio.imread('noise0' + '\\' + filename)
                mses[i] += mse(clear_img, gaussian_filter(noisy_img, sigmas[i], mode='nearest'))
        params[num] = sigmas[np.argmin(mses)]
    interpolator = interp1d(noises, params, kind='linear')
    return interpolator([noise_level])[0]


def denoise(img):
    noise = img - gaussian_filter(img, 1, mode='nearest')
    noise_level = np.std(noise)
    sigma = query(noise_level)
    return gaussian_filter(img, sigma=sigma, mode='nearest')


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

parser_mse = subparsers.add_parser('mse', help='Calculates MSE \
    between two images')
parser_mse.add_argument('input_file_1', type=str)
parser_mse.add_argument('input_file_2', type=str)

parser_calc_noise = subparsers.add_parser('calc_noise', help='Calculates \
    standard deviation of the noise')
parser_calc_noise.add_argument('input_file', type=str)

parser_median = subparsers.add_parser('median', help='Median filter')
parser_median.add_argument('r', type=int)
parser_median.add_argument('input_file', type=str)
parser_median.add_argument('output_file', type=str)

parser_query = subparsers.add_parser('query', help='Print noise reduction \
    options')
parser_query.add_argument('noise_level', type=float)

parser_denoise = subparsers.add_parser('denoise', help='Noise reduction')
parser_denoise.add_argument('input_file', type=str)
parser_denoise.add_argument('output_file', type=str)

args = parser.parse_args()
if args.command == 'mse':
    img1 = imageio.imread(args.input_file_1).astype(np.float64)
    img2 = imageio.imread(args.input_file_2).astype(np.float64)
    if img1.ndim == 3:
        img1 = make_gray_scale(img1)
    if img2.ndim == 3:
        img2 = make_gray_scale(img2)
    res = mse(img1, img2)
    if res is not None:
        print('MSE: ' + str(res))

elif args.command == 'calc_noise':
    img = imageio.imread(args.input_file).astype(np.float64)
    print('Standard deviation of the noise: ' + str(calc_noise(img)))

elif args.command == 'median':
    img = imageio.imread(args.input_file).astype(np.float64)
    if img.ndim == 3:
        img = make_gray_scale(img)
    res = median(img, args.r)
    imageio.imwrite(args.output_file, res.astype(np.uint8))

elif args.command == 'query':
    sigma_d = query(args.noise_level)
    print(0, sigma_d, 0, sep=' ')

elif args.command == 'denoise':
    img = imageio.imread(args.input_file).astype(np.float64)
    res = denoise(img)
    imageio.imwrite(args.output_file, res.astype(np.uint8))
