import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cPickle as pickle

from sklearn import externals as ext
from skimage import io
from skimage import segmentation as seg
from skimage.future import graph
from pystruct import utils as psutil

import features as feat
import utils as util
import crfrnn.fg_prediction as fg


PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
LABEL_DICT_FILE = os.path.join(PROJECT_DIR, 'data/raw/ss/cat/ss_10_cats.csv')


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--unary-model-path', help='unary model filename')
    parser.add_argument('--crf-model-path', help='unary model filename')
    parser.add_argument('--image-path', help='image filename')
    args = parser.parse_args()

    # # read image and compute superpixels
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
    unary_features = feat.compute_unary_features(image, sps, gabor_kernels, face_detectors, fg_mask)
    if unary_features is None:
        print "face is not detected"
        exit()

    # load unary model
    (unary_scaler, unary_clf) = ext.joblib.load(args.unary_model_path)

    # compute crf node features from unary potentials
    scaled_unary_features = unary_scaler.transform(unary_features)
    node_features = unary_clf.predict_proba(scaled_unary_features)

    # load edge features scaler
    ef_scaler = pickle.load(open(args.crf_model_path.split('.')[0] + '_scaler.pkl', 'rb'))

    # generate edges
    # vertices, edges = feat.generate_edges(sps)
    edges = np.array(graph.rag_mean_color(image, sps).edges())

    # extract edge features
    edge_features = feat.compute_crf_edge_features(unary_features, edges)
    edge_features = ef_scaler.transform(edge_features)

    # build data test
    X_test = [(node_features, edges, edge_features)]

    # load crf model from pystruct
    logger = psutil.SaveLogger(args.crf_model_path)
    crf_clf = logger.load()

    # predict
    labels = crf_clf.predict(X_test)[0]


    # load label
    true_labels = util.get_label_dictionary(label_dict_file=LABEL_DICT_FILE, fileformat='str')

    # get unique labels
    unique_labels = np.unique(np.hstack(labels))

    # filter label do not include background, skin, and hair
    unique_labels = [l for l in unique_labels if l > 2]

    # get all segment mask
    all_mask = image.copy()
    for i, segment in enumerate(np.unique(sps)):
        mask = (sps == segment).reshape(all_mask.shape[0], all_mask.shape[1], 1).repeat(3, axis=2)
        all_mask[mask] = labels[i]

    # get all segment mask per clothing category
    segment_masks = []
    output = np.zeros(image.shape)
    rectangles = []
    rectangle_labels = []
    for label in unique_labels:
        segment_mask = (all_mask == label)
        segment_masks.append(segment_mask)

        # for visualization, set color and label
        color = np.zeros(3) if label == 0 else np.random.randint(0, 256, size=(3))
        output[segment_mask[:,:,0],:] = color
        rectangle = mpatches.Rectangle((0, 0), 1, 1, fc=tuple(color / 255.0))
        rectangles.append(rectangle)
        rectangle_labels.append(str(true_labels[str(label)]))

    # visualization
    n_axis = 2 + len(unique_labels)
    fig = plt.figure("vis")

    ax1 = fig.add_subplot(1, n_axis, 1)
    ax1.set_title("original image")
    ax1.imshow(image)

    ax2 = fig.add_subplot(1, n_axis, 2)
    ax2.set_title("segmented")
    ax2.imshow(image)
    ax2.imshow(output, cmap='jet', alpha=0.7)

    for i, label in enumerate(unique_labels):

        cropped = np.zeros(image.shape, dtype='uint8')
        cropped[segment_masks[i]] = image[segment_masks[i]]

        ax3 = fig.add_subplot(1, n_axis, i+3)
        ax3.set_title(true_labels[str(label)])
        ax3.imshow(cropped)

    plt.show()
