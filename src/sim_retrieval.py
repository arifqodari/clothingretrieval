import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from skimage import io
from skimage import transform as tran
from skimage import color as color
from sklearn import neighbors as neigh

import features as feat


DEFAULT_MAX_WIDTH = 100


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', help='features filename')
    parser.add_argument('--image-query', help='image query path')
    args = parser.parse_args()

    # load features
    (tree, image_paths) = pickle.load(open(args.features, 'rb'))

    # prepare gabor kernels
    gabor_kernels = feat.prepare_gabor_kernels(gabor_freqs=[0.2])

    # load image
    extension = "." + args.image_query.split(".")[-1]
    crop_path = args.image_query.split(extension)[0] + "_crop" + extension
    image = io.imread(crop_path)

    # compute scale factor
    h, w, d = image.shape
    scale = float(DEFAULT_MAX_WIDTH) / w

    # resize image
    image = tran.rescale(image, scale=(scale, scale))
    image_gray = color.rgb2gray(image)

    # load mask
    mask_path = args.image_query.split(extension)[0] + "_mask" + extension
    mmask = io.imread(mask_path)
    mmask = tran.rescale(mmask, scale=(scale, scale))
    mask = np.zeros(mmask.shape, dtype='uint8')
    mask[mmask > 0] = 255
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1).repeat(3, axis=2)

    # get gabor response from the whole image
    gabor_responses = feat.compute_gabor_responses(image_gray, gabor_kernels)

    # extract features from test image
    query_features = feat.sim_feature_extraction(image, gabor_responses, mask)

    # build KD Tree
    K = 10
    dists, indices = tree.query(query_features.reshape(1, -1), k=K)

    # visualize query image
    fig = plt.figure("vis")
    ax0 = fig.add_subplot(1, 11, 1)
    ax0.imshow(image)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("query image")

    # visualize results
    for i, idx in enumerate(indices[0]):
        im = io.imread(image_paths[idx]['full'])
        ax = fig.add_subplot(1, 11, i+2)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("result #%i" % (i+1))
    plt.show()
