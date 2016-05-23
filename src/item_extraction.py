import os
import argparse
import numpy as np

from skimage import io

import features as feat


def extract(photo_dir, anno_dir, cat_id=0):
    """
    load annotation file and return the target
    """

    # get list of filenames
    filenames = os.listdir(anno_dir)
    targets = []
    images = []

    for i, target_filename in enumerate(filenames):

        image_filename = target_filename.split('.')[0] + '.jpg'
        image_path = os.path.join(photo_dir, image_filename)
        target_path = os.path.join(anno_dir, target_filename)
        if (not os.path.isfile(target_path)) or (not os.path.isfile(image_path)):
            continue

        # load image
        image = io.imread(image_path)

        # load pixel-wise annotation
        pixel_class = feat.load_pixel_annotations(target_path).reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)

        # masking
        image[pixel_class != cat_id] = 0
        if image.sum() == 0:
            continue

        # save the file
        io.imsave(os.path.join('data/interim', image_filename), image)


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--photo-dir', help='photo directories')
    parser.add_argument('--anno-dir', help='anno directories')
    parser.add_argument('--cat', help='category')
    args = parser.parse_args()

    # extract
    extract(args.photo_dir, args.anno_dir, int(args.cat))
