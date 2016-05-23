import os
import numpy as np
import cPickle as pickle
import argparse

from skimage import io
from skimage import transform as tran
from skimage import color as color
from sklearn import neighbors as neigh
from sys import argv

import features as feat
import utils as util


DEFAULT_MAX_WIDTH = 100


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='image directory')
    parser.add_argument('--output-dir', help='output directory')
    args = parser.parse_args()

    # prepare gabor kernels
    gabor_kernels = feat.prepare_gabor_kernels(gabor_freqs=[0.2])

    # get image paths
    image_paths = util.get_image_paths(args.image_dir)

    all_features = []
    for i, image_path in enumerate(image_paths):
        print "%i from %i" % (i + 1, len(image_paths))

        # load images
        image = io.imread(image_path['crop'])

        # compute scale factor
        h, w, d = image.shape
        scale = float(DEFAULT_MAX_WIDTH) / w

        # rescale image
        image = tran.rescale(image, scale=(scale, scale))
        image_gray = color.rgb2gray(image)

        # load and rescale mask
        mmask = io.imread(image_path['mask'])
        mmask = tran.rescale(mmask, scale=(scale, scale))
        mask = np.zeros(mmask.shape, dtype='uint8')
        mask[mmask > 0] = 255
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1).repeat(3, axis=2)

        # get gabor response from the whole image
        gabor_responses = feat.compute_gabor_responses(image_gray, gabor_kernels)

        # extract features
        features = feat.sim_feature_extraction(image, gabor_responses, mask)

        all_features.append(features)

    # build KDTree
    tree = neigh.KDTree(all_features)

    # save the extracted features
    pickle.dump((tree, image_paths), open(os.path.join(args.output_dir, 'sim_features.pkl'), 'wb'))
