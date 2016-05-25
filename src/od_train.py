import os
import argparse
import dlib
import numpy as np
import cv2

from skimage import io
from sklearn import cross_validation as crosval

import features as feat


def load_data(photo_dir, anno_dir, bg_cat_id=0):
    """
    load annotation file and return the target
    """

    # get list of filenames
    filenames = os.listdir(anno_dir)
    targets = []
    images = []

    for i, target_filename in enumerate(filenames):

        image_path = os.path.join(photo_dir, target_filename.split('.')[0] + '.jpg')
        target_path = os.path.join(anno_dir, target_filename)
        if (not os.path.isfile(target_path)) or (not os.path.isfile(image_path)):
            continue

        # load pixel-wise annotation
        pixel_class = feat.load_pixel_annotations(target_path)

        # create label mask
        mask = np.ones(pixel_class.shape, dtype='uint8') * 255
        mask[pixel_class == bg_cat_id] = 0
        io.imsave('data/interim/' + target_filename + '.jpg', mask)

        # get contours
        edges = cv2.Canny(mask, 50, 100)
        contours, _ = cv2.findContours(edges, mode=cv2.cv.CV_RETR_TREE, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        # get bounding rectangle
        x,y,w,h = cv2.boundingRect(np.vstack(contours))
        rect = [dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)]

        targets.append(rect)
        images.append(io.imread(image_path))

    return images, targets


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--photo-dirs', metavar='photo_dirs', nargs='*', help='input directories')
    parser.add_argument('--anno-dirs', metavar='anno_dirs', nargs='*', help='input directories')
    parser.add_argument('--model-filename', help='model filename')
    args = parser.parse_args()

    # load data
    X, y = [], []
    for i in xrange(0, len(args.photo_dirs)):
        XX, yy = load_data(args.photo_dirs[i], args.anno_dirs[i])
        X += XX
        y += yy

    # split the data
    test_ratio = 0.1
    X_train, X_test, y_train, y_test = crosval.train_test_split(X, y, test_size=test_ratio, random_state=0)

    # set training options
    options = dlib.simple_object_detector_training_options()
    options.C = 3
    options.num_threads = 6
    options.be_verbose = True

    # perform training
    detector = dlib.train_simple_object_detector(X_train, y_train, options)

    # save the model
    detector.save(args.model_filename)

    # testing
    print("")
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(X_train, y_train, detector)))
    print("Testing accuracy: {}".format(
        dlib.test_simple_object_detector(X_test, y_test, detector)))
