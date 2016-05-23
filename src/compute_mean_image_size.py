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

    mean_shape = np.zeros(3)
    min_max_height = np.array((9999, -9999))
    min_max_width = np.array((9999, -9999))
    n = 0

    for i, filename in enumerate(filenames):
        image_path = os.path.join(args.image_dir, filename)
        if os.path.isfile(image_path):
            image = io.imread(image_path)
            mean_shape += np.array(image.shape)
            min_max_height[0] = min(image.shape[0], min_max_height[0])
            min_max_height[1] = max(image.shape[0], min_max_height[1])
            min_max_width[0] = min(image.shape[1], min_max_width[0])
            min_max_width[1] = max(image.shape[1], min_max_width[1])
            n += 1

    mean_shape /= float(n)
    print 'min max height'
    print np.ceil(min_max_height).astype(int)
    print 'min max width'
    print np.ceil(min_max_width).astype(int)
    print 'average image size'
    print np.ceil(mean_shape).astype(int)
