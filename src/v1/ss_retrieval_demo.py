import os
import sys
import numpy as np
import argparse
from skimage import io
from features import SimilarityFeaturesCollection
from features_extraction import SimilarityFeaturesExtractor


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='input image directory')
    parser.add_argument('--query-image', help='query image path')
    args = parser.parse_args()
    return args.image_dir, args.query_image


def features_extraction(images_dir):
    filenames = os.listdir(images_dir)
    feat_extractor = SimilarityFeaturesExtractor()
    features_set = SimilarityFeaturesCollection()

    for i, filename in enumerate(filenames):
        image_path = os.path.join(images_dir, filename)
        if not os.path.isfile(image_path):
            continue

        image = io.imread(image_path)
        features = feat_extractor.extract(image)
        features_set.append(features, filename)

    features_set.init_kd_tree()
    return features_set


def get_query_features(image_path):
    query_image = io.imread(image_path)
    feat_extractor = SimilarityFeaturesExtractor()
    return feat_extractor.extract(query_image)


def main():
    image_dir, query_image_path = argument_parser()

    features_set = features_extraction(image_dir)
    query_features = get_query_features(query_image_path)
    similar_product_filenames = features_set.search(query_features)
    print similar_product_filenames


if __name__ == '__main__':
    main()
