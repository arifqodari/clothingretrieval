import argparse
import os
import numpy as np

from skimage import io


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='image directory')
    args = parser.parse_args()

    # get list of image files
    filenames = os.listdir(args.image_dir)

    mean_rgb = np.zeros(3)
    n = 0

    for i, filename in enumerate(filenames):
        # get image path
        image_path = os.path.join(args.image_dir, filename)

        if os.path.isfile(image_path):

            image = io.imread(image_path)

            mean_rgb += image.sum(axis=(0,1))
            n += (image.shape[0] * image.shape[1])

    mean_rgb /= float(n)
    print 'average image size'
    print mean_rgb
