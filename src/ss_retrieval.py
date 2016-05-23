import os
import numpy as np
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt

from skimage import io
from skimage import transform as tran
from skimage import color as cl
from skimage import util as skutil
from skimage.future import graph
from sklearn import externals as ext
from sklearn import neighbors as neigh
from pystruct import utils as psutil

import features as feat
import utils as util


PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
LABEL_DICT_FILE = os.path.join(PROJECT_DIR, 'data/raw/ss/cat/ss_10_cats.csv')


def feature_extraction(image, mask):
    """
    similarity feature extraction
    """

    image_gray = cl.rgb2gray(skutil.img_as_float(image))

    # get gabor response from the whole image
    gabor_responses = feat.compute_gabor_responses(image_gray, gabor_kernels)

    # extract features
    return feat.sim_feature_extraction(image, gabor_responses, mask)


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--unary-model-path', help='unary model filename')
    parser.add_argument('--crf-model-path', help='unary model filename')
    parser.add_argument('--cat', help='category')
    parser.add_argument('--features', help='features file')
    parser.add_argument('--image-path', help='image path')
    args = parser.parse_args()

    # prepare gabor kernels
    gabor_kernels = feat.prepare_gabor_kernels(gabor_freqs=[0.2])

    # prepare face detectors
    face_detectors = feat.prepare_face_detectors()

    # load label dictionary
    label_dict = util.get_label_dictionary(label_dict_file=LABEL_DICT_FILE, fileformat='str')
    cat_id = int([key for key in label_dict if label_dict[key] == args.cat][0])

    # load image
    image = io.imread(args.image_path)

    # extract superpixels
    sps = feat.compute_superpixels(image, 300)

    # extract unary features from test image
    unary_features = feat.compute_unary_features(image, sps, gabor_kernels, face_detectors)

    # compute mask
    if unary_features is None:
        active_mask = feat.get_default_mask(image)
    else:
        active_mask = feat.get_semantic_segmentation_mask(image, args.unary_model_path, sps, unary_features, args.crf_model_path, cat_id)

    # feature extraction
    features = feat.compute_sim_features(image, active_mask)

    # load features
    (tree, image_paths) = pickle.load(open(args.features, 'rb'))

    # search similar
    K = 2
    dists, indices = tree.query(features.reshape(1, -1), k=K)

    # visualize query image
    fig = plt.figure("vis")
    ax0 = fig.add_subplot(1, K+2, 1)
    ax0.imshow(image)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("query image")

    ax1 = fig.add_subplot(1, K+2, 2)
    ax1.imshow(image)
    ax1.imshow(active_mask, cmap='jet', alpha=0.7)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("segmentation")

    # visualize results
    for i, idx in enumerate(indices[0]):
        im = io.imread(image_paths[idx])
        ax = fig.add_subplot(1, K+2, i+3)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("similar result #%i" % (i+1))
    plt.show()
