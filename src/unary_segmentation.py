import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io
from sklearn import externals as ext
from sklearn import preprocessing as preproc

import features as feat
import utils as util
import crfrnn.fg_prediction as fg


PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
LABEL_DICT_FILE = os.path.join(PROJECT_DIR, 'data/raw/ss/cat/ss_10_cats.csv')


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='model filename')
    parser.add_argument('--image-path', help='image filename')
    args = parser.parse_args()

    # read image and compute superpixels
    # image = io.imread(args.image_path)
    # sps = feat.compute_superpixels(image, 300)

    # prepare gabor kernels
    gabor_kernels = feat.prepare_gabor_kernels(gabor_freqs=[0.2])

    # prepare face detectors
    face_detectors = feat.prepare_face_detectors()

    # mask
    image, fg_mask = fg.predict(args.image_path, os.path.join(PROJECT_DIR, 'src/crfrnn/TVG_CRFRNN_COCO_VOC.caffemodel'), os.path.join(PROJECT_DIR, 'src/crfrnn/TVG_CRFRNN_new_deploy.prototxt'))

    # compute superpixels
    sps = feat.compute_superpixels(image, 300)

    # extract features from test image
    features = feat.compute_unary_features(image, sps, gabor_kernels, face_detectors, fg_mask)
    if features is None:
        print "face is not detected"
        exit()


    # load unary model
    (scaler, clf) = ext.joblib.load(args.model_path)

    # preprocess data
    features = scaler.transform(features)

    # predict the labels
    labels = clf.predict(features)

    # load label
    true_labels = util.get_label_dictionary(label_dict_file=LABEL_DICT_FILE, fileformat='str')

    # generate colors for labels
    colors = {}
    rectangles = []
    rectangle_labels = []
    for (i, label) in enumerate(np.unique(np.hstack(labels))):
        if label == 0:
            colors[0] = np.zeros(3)
        else:
            colors[label] = np.random.randint(0, 256, size=(3))

        rectangle = mpatches.Rectangle((0, 0), 1, 1, fc=tuple(colors[label] / 255.0))
        rectangles.append(rectangle)
        rectangle_labels.append(str(true_labels[str(label)]))

    # create segmentation output
    output = image.copy()
    for (i, segment) in enumerate(np.unique(sps)):
        mask = (sps == segment).reshape(output.shape[0], output.shape[1], 1).repeat(3, axis=2)
        for c in xrange(0, 3):
            output[mask[:,:,c], c] = colors[labels[i]][c]
            output[fg_mask == 0, c] = 0

    # visualization
    fig = plt.figure("vis")
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("original image")
    ax1.imshow(image)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("segmented")
    ax2.imshow(image)
    ax2.imshow(output, cmap='jet', alpha=0.7)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("segmented")
    ax3.imshow(output)

    plt.legend(rectangles, rectangle_labels, loc='best', bbox_to_anchor=(1.4,0.8))

    plt.show()
