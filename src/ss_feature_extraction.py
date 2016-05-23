import os
import numpy as np
import argparse
import cPickle as pickle

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
    parser.add_argument('--inp-dir', help='input directory')
    parser.add_argument('--out-dir', help='output directory')
    args = parser.parse_args()

    # get list of all image filenames
    filenames = os.listdir(args.inp_dir)

    # prepare gabor kernels
    gabor_kernels = feat.prepare_gabor_kernels(gabor_freqs=[0.2])

    # prepare face detectors
    face_detectors = feat.prepare_face_detectors()

    # load label dictionary
    label_dict = util.get_label_dictionary(label_dict_file=LABEL_DICT_FILE, fileformat='str')
    cat_id = int([key for key in label_dict if label_dict[key] == args.cat][0])

    all_features = []
    image_paths = []
    for i, filename in enumerate(filenames):
        print "%i from %i" % (i+1, len(filenames))

        image_path = os.path.join(args.inp_dir, filename)
        if not os.path.isfile(image_path):
            continue

        # load image
        image = io.imread(image_path)

        # extract superpixels
        sps = feat.compute_superpixels(image, 300)

        # extract unary features from test image
        unary_features = feat.compute_unary_features(image, sps, gabor_kernels, face_detectors)

        # compute mask
        if unary_features is None:
            active_mask = feat.get_default_mask(image)
        else:
            active_mask = feat.get_semantic_segmentation_mask(image, args.unary_model_path, sps, unary_features, args.crf_model_path, cat_id)

        # ignore if the resulting mask is empty
        if active_mask.sum() == 0:
            continue

        # feature extraction
        features = feat.compute_sim_features(image, active_mask)

        # add features and image path
        all_features.append(features)
        image_paths.append(image_path)

        # for debugging purpose
        im = image.copy()
        im[active_mask == 0] = 0
        io.imsave(os.path.join('data/interim/', filename), im)

    # build KDTree
    tree = neigh.KDTree(all_features)

    # save the extracted features
    pickle.dump((tree, image_paths), open(os.path.join(args.out_dir, 'sim_features_' + args.cat + '.pkl'), 'wb'))
