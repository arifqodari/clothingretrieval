import os
import argparse
import numpy as np

from skimage import io

import features as feat
import utils as util


def unary_feature_extraction(anno_dir, photo_dir, output_dir, label_dict_file=None):
    """
    features extraction to train unary potential
    """

    # prepare gabor kernels
    gabor_kernels = feat.prepare_gabor_kernels(gabor_freqs=[0.2])

    # prepare face detectors
    face_detectors = feat.prepare_face_detectors()

    # label dictionary
    label_dict = util.get_label_dictionary(label_dict_file)

    # get all csv annotation files
    csv_dirs = os.listdir(anno_dir)

    X, y = np.array([]), np.array([])
    for i, csv_file in enumerate(csv_dirs):
        print "%i from %i" % (i, len(csv_dirs))

        csv_path = os.path.join(anno_dir, csv_file)
        image_path = os.path.join(photo_dir, csv_file.split(".")[0] + ".jpg")

        if os.path.isfile(csv_path):

            image = io.imread(image_path)
            sps = feat.compute_superpixels(image, 300)

            pixel_class = feat.load_pixel_annotations(csv_path)
            unary_features = feat.compute_unary_features(image, sps, gabor_kernels, face_detectors, pixel_class)

            if unary_features is not None:

                # generate labels
                labels = feat.generate_labels(pixel_class, sps, label_dict)

                X = np.vstack([X, unary_features]) if X.size else unary_features
                y = np.append(y, labels) if y.size else labels

    # save the features
    part_idx = anno_dir.split("/")[-2]
    np.save(os.path.join(output_dir, 'X_unary_part_' + part_idx + '.npy'), X)
    np.save(os.path.join(output_dir, 'y_unary_part_' + part_idx + '.npy'), y)




if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-dir', help='csv_parts annotation directory')
    parser.add_argument('--photo-dir', help='photo directory')
    parser.add_argument('--output-dir', help='output directory')
    parser.add_argument('--label-dict-file', help='label dictionary file')
    args = parser.parse_args()

    # perform feature extraction
    unary_feature_extraction(args.anno_dir, args.photo_dir, args.output_dir, args.label_dict_file)
