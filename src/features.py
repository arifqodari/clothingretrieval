import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy import ndimage as ndi
from skimage import io
from skimage import segmentation as seg
from skimage import filters as filters
from skimage import util as util
from skimage import color as color
from skimage.future import graph
from skimage.feature import local_binary_pattern
from sklearn import externals as ext
from pystruct import utils as psutil


DEFAULT_N_SEGMENTS = 500
DEFAULT_SIGMA = 5
DEFAULT_MAX_GABOR_THETA = 4
DEFAULT_GABOR_SIGMAS = (1,2)
DEFAULT_GABOR_FREQS = (0.05, 0.25)
DEFAULT_MIN_FACE = 50
DEFAULT_FACE_ID = 0


PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
DEFAULT_FFACE_DETECTOR_MODEL = os.path.join(PROJECT_DIR, 'models/face/haarcascade_frontalface_default.xml')
DEFAULT_PFACE_DETECTOR_MODEL = os.path.join(PROJECT_DIR, 'models/face/haarcascade_profileface.xml')


def prepare_gabor_kernels(max_gabor_theta=DEFAULT_MAX_GABOR_THETA, gabor_sigmas=DEFAULT_GABOR_SIGMAS, gabor_freqs=DEFAULT_GABOR_FREQS):
    """
    prepare gabor kernel to capture texture
    """
    kernels = []
    for theta in range(max_gabor_theta):
        theta = theta / float(max_gabor_theta) * np.pi
        for sigma in gabor_sigmas:
            for frequency in gabor_freqs:
                kernel = np.real(filters.gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels


def prepare_face_detectors(fface_detector=DEFAULT_FFACE_DETECTOR_MODEL, pface_detector=DEFAULT_PFACE_DETECTOR_MODEL):
    """
    prepare face detector model
    frontal first then profile
    """
    return [cv2.CascadeClassifier(fface_detector), cv2.CascadeClassifier(pface_detector)]


def compute_face_center(face, norm=None):
    """
    get (normalized) coordinate center of the given face
    """
    face_center = [face[0] + (face[2] * 0.5), face[1] + (face[3] * 0.5)]
    if norm is not None:
        return [face_center[0] / norm[0], face_center[1] / norm[1]]
    else:
        return face_center


def check_face_true_positive(face, pixel_annotations):
    """
    check if given face has positive label / not a background
    """

    face_center = compute_face_center(face)
    if pixel_annotations[int(face_center[1]), int(face_center[0])] > 0:
        return face


def check_foreground(pixel_annotations, segment_mask):
    """
    check if given superpixel region has positive label / not a background
    """

    return np.all(pixel_annotations[segment_mask])


def detect_face(image, face_detectors, pixel_annotations=None):
    """
    detect face and return location of detected faces
    """
    for face_detector in face_detectors:
        faces = face_detector.detectMultiScale(image, 1.3, 5, minSize=(DEFAULT_MIN_FACE,DEFAULT_MIN_FACE))

        if len(faces) > 0:
            if pixel_annotations is not None:
                return check_face_true_positive(faces[DEFAULT_FACE_ID], pixel_annotations)
            else:
                return faces[DEFAULT_FACE_ID]


def compute_superpixels(image, n_segments=DEFAULT_N_SEGMENTS):
    """
    SLIC-based superpixels
    """
    return seg.slic(image, n_segments = n_segments, sigma = DEFAULT_SIGMA)


def compute_segment_center(gridx, gridy, mask, norm=None):
    """
    get (normalized) 2D coordinate of the center of the masked image
    """
    segment_center = [gridx[mask[:,:,0]].mean(), gridy[mask[:,:,0]].mean()]
    if norm is not None:
        return [segment_center[0] / norm[0], segment_center[1] / norm[1]]
    else:
        return segment_center


def compute_histogram_color(image, mask, color='rgb'):
    """
    compute color histogram each channel separately
    """

    if color == 'rgb':
        ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    elif color == 'lab':
        ranges = [[0.0, 100.0], [-127.0, 127.0], [-127.0, 127.0]]

    histogram = np.array([])
    for i in xrange(0, 3):
        hist = cv2.calcHist([image], [i], mask[:,:,0], [10], ranges[i])
        histogram = np.append(histogram, cv2.normalize(hist, norm_type=cv2.NORM_MINMAX)) if histogram.size else cv2.normalize(hist, norm_type=cv2.NORM_MINMAX)
    return histogram


def compute_gabor_responses(image, kernels):
    """
    compute gabor responses
    """
    responses = []
    for kernel in kernels:
        filtered = ndi.convolve(image, kernel, mode='wrap')
        responses.append(filtered.astype('float32'))
    return responses


def compute_histogram_gabor(gabor_responses, mask):
    """
    get gabor features (mean and std) from a set of gabor responses
    """
    histogram = np.array([])
    for response in gabor_responses:
        hist = cv2.calcHist([response], [0], mask[:,:,0], [10], [-1.0, 1.0])
        histogram = np.append(histogram, cv2.normalize(hist, norm_type=cv2.NORM_MINMAX)) if histogram.size else cv2.normalize(hist, norm_type=cv2.NORM_MINMAX)
    return histogram


def compute_face_offset(face_center, segment_center):
    """
    compute offset vector from face center to the segment center
    """
    return [face_center[0] - segment_center[0], face_center[1] - segment_center[1]]


def load_pixel_annotations(csv_path):
    """
    load pixel-level annotations
    """
    return np.loadtxt(csv_path, delimiter=',', dtype='int')


def generate_labels(pixel_annotations, sps, label_dict=None):
    labels = []
    for (i, segment) in enumerate(np.unique(sps)):
        # get segment class
        arr = pixel_annotations[sps == segment].flatten()

        # get segment class
        if label_dict is not None:
            segment_class = label_dict[np.argmax(np.bincount(arr))]
        else:
            segment_class = np.argmax(np.bincount(arr))

        # concatentate label
        labels.append(segment_class)

    return np.array(labels)


def compute_node_features(image, sps, gabor_kernels, face_detectors, pixel_annotations=None):
    """
    get node features from each segment in superpixels
    """

    # detect face on the image, if there is no face discard it
    face = detect_face(image, face_detectors, pixel_annotations)
    if face is None:
        return None
    else:
        # normalized face center
        face_center = compute_face_center(face, sps.shape[::-1])

    # get grids from superpixel
    gridy, gridx = np.mgrid[:sps.shape[0], :sps.shape[1]]

    # convert image to different color spaces
    image_rgb = util.img_as_float(image)
    image_rgb = cv2.GaussianBlur(image_rgb, (5,5), 0)
    image_rgb32 = image_rgb.astype('float32')
    image_lab = color.rgb2lab(image_rgb)
    image_lab32 = image_lab.astype('float32')
    image_gray = color.rgb2gray(image_rgb)

    # get gabor response from the whole image
    gabor_responses = compute_gabor_responses(image_gray, gabor_kernels)

    features = np.array([])
    segments = np.unique(sps)
    for segment in segments:
        node_features = []

        # create mask image
        mask = (sps == segment).reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)
        mask8 = util.img_as_ubyte(mask)

        # information foreground or background
        fg = check_foreground(pixel_annotations, mask[:,:,0]).astype('int')
        node_features = np.append(node_features, fg)

        # histogram of rgb value
        hist_rgb = compute_histogram_color(image_rgb32, mask8)
        node_features = np.append(node_features, hist_rgb)

        # histogram of lab value
        hist_lab = compute_histogram_color(image_lab32, mask8, color='lab')
        node_features = np.append(node_features, hist_lab)

        # normalized 2D coordinate
        segment_center = compute_segment_center(gridx, gridy, mask, sps.shape[::-1])
        node_features = np.append(node_features, segment_center)

        # normalized 2D offset from detected face
        face_offset = compute_face_offset(segment_center, face_center)
        node_features = np.append(node_features, face_offset)

        # gabor filter response
        segment_responses = compute_histogram_gabor(gabor_responses, mask8)
        node_features = np.append(node_features, segment_responses)

        # concatenate all node features
        features = np.vstack([features, node_features]) if features.size else node_features

    return features


def generate_edges(sps):
    """
    generate edges from superpixels
    """
    edges = psutil.make_grid_edges(sps)
    vertices = np.unique(sps)
    n_vertices = vertices.shape[0]

    # filter out edges that connect to themselves
    crossings = edges[sps.ravel()[edges[:, 0]] != sps.ravel()[edges[:, 1]]]
    edges = sps.ravel()[crossings]
    edges = np.sort(edges, axis=1)

    # find unique crossing
    crossing_hash = (edges[:, 0] + n_vertices * edges[:, 1])
    unique_hash = np.unique(crossing_hash)
    unique_crossings = np.c_[unique_hash % n_vertices, unique_hash // n_vertices]

    return vertices, unique_crossings


def compute_edge_features(node_features, edges):
    """
    compute edge features
    """
    edge_features = np.array([])

    for edge in edges:
        # color (rgb and lab) differences
        # edge_feature = np.linalg.norm(node_features[edge[1], 0:60] - node_features[edge[0], 0:60])

        # foreground feature
        edge_feature = node_features[0]

        # color differences
        color_diff = np.linalg.norm(node_features[edge[1], 1:61] - node_features[edge[0], 1:61])
        edge_feature = np.append(edge_feature, color_diff)

        # gabor filter differences
        # gabor_diff = np.linalg.norm(node_features[edge[1], 64:] - node_features[edge[0], 64:])
        gabor_diff = np.linalg.norm(node_features[edge[1], 65:] - node_features[edge[0], 65:])
        edge_feature = np.append(edge_feature, gabor_diff)

        # relative position
        # pos_offset = node_features[edge[1], 60:62] - node_features[edge[0], 60:62]
        pos_offset = node_features[edge[1], 61:63] - node_features[edge[0], 61:63]
        edge_feature = np.append(edge_feature, pos_offset)

        edge_features = np.vstack([edge_features, edge_feature]) if edge_features.size else edge_feature

    return edge_features


def compute_unary_features(image, sps, gabor_kernels, face_detectors, pixel_annotations=None):
    return compute_node_features(image, sps, gabor_kernels, face_detectors, pixel_annotations)


def compute_crf_edge_features(node_features, edges):
    return compute_edge_features(node_features, edges)


def compute_histogram_color_combined(image, mask, color='rgb'):
    """
    compute 3D color histogram
    """

    if color == 'rgb':
        ranges = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    elif color == 'lab':
        ranges = [0.0, 100.0, -127.0, 127.0, -127.0, 127.0]

    hist = cv2.calcHist([image], [0, 1, 2], mask[:,:,0], [8, 8, 8], ranges)
    return cv2.normalize(hist, norm_type=cv2.NORM_MINMAX).flatten()


def sim_feature_extraction(image, gabor_responses, mask):
    """
    feature extraction for similarity
    """

    image_rgb32 = image.astype('float32')
    image_lab32 = color.rgb2lab(image).astype('float32')

    rgb_histogram = compute_histogram_color_combined(image_rgb32, mask)
    lab_histogram = compute_histogram_color_combined(image_lab32, mask, color='lab')
    gabor_features = compute_histogram_gabor(gabor_responses, mask)

    return np.hstack([rgb_histogram, lab_histogram, gabor_features])


def get_default_mask(image):
    """
    use the entine image for mask
    """

    # convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)

    # get contours
    edges = cv2.Canny(image_gray, 50, 100)
    contours, _ = cv2.findContours(edges, mode=cv2.cv.CV_RETR_TREE, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)

    # get convex hull
    hull = [cv2.convexHull(np.vstack(contours))]
    image_mask = np.zeros(image_gray.shape, 'uint8')
    cv2.drawContours(image_mask, hull, contourIdx=-1, color=255, thickness=cv2.cv.CV_FILLED)

    return image_mask.reshape(image_mask.shape[0], image_mask.shape[1], 1).repeat(3, axis=2)


def get_unary_mask(image, unary_model_path, sps, unary_features, cat_id):
    """
    perform semantic segmentation and return mask
    """

    # load unary model and scaler
    (unary_scaler, unary_clf) = ext.joblib.load(unary_model_path)

    # compute crf node features from unary potentials
    scaled_unary_features = unary_scaler.transform(unary_features)

    # predict final label
    labels = unary_clf.predict(scaled_unary_features)

    # get active mask based on given category id
    active_mask = np.zeros(sps.shape, dtype='uint8')
    for i, segment in enumerate(np.unique(sps)):
        if labels[i] == cat_id:
            mask = (sps == segment)
            active_mask[mask] = 255

    return active_mask.reshape(active_mask.shape[0], active_mask.shape[1], 1).repeat(3, axis=2)


def get_crf_mask(image, unary_model_path, sps, unary_features, crf_model_path, cat_id):
    """
    perform semantic segmentation and return mask
    """

    # load unary model and scaler
    (unary_scaler, unary_clf) = ext.joblib.load(unary_model_path)

    # compute crf node features from unary potentials
    scaled_unary_features = unary_scaler.transform(unary_features)
    node_features = unary_clf.predict_proba(scaled_unary_features)

    # load edge features scaler
    ef_scaler = pickle.load(open(crf_model_path.split('.')[0] + '_scaler.pkl', 'rb'))

    # generate edges
    edges = np.array(graph.rag_mean_color(image, sps).edges())

    # extract edge features
    edge_features = compute_crf_edge_features(unary_features, edges)
    edge_features = ef_scaler.transform(edge_features)

    # build data test
    X_test = [(node_features, edges, edge_features)]

    # load crf model from pystruct
    logger = psutil.SaveLogger(crf_model_path)
    crf_clf = logger.load()

    # predict final label
    labels = crf_clf.predict(X_test)[0]

    # get active mask based on given category id
    active_mask = np.zeros(sps.shape, dtype='uint8')
    for i, segment in enumerate(np.unique(sps)):
        if labels[i] == cat_id:
            mask = (sps == segment)
            active_mask[mask] = 255

    return active_mask.reshape(active_mask.shape[0], active_mask.shape[1], 1).repeat(3, axis=2)


def get_semantic_segmentation_mask(image, unary_model_path, sps, unary_features, crf_model_path, cat_id):
    return get_unary_mask(image, unary_model_path, sps, unary_features, cat_id)
    # return get_crf_mask(image, unary_model_path, sps, unary_features, crf_model_path, cat_id)


def compute_lbp_features(image, mask, radius=2):
    """
    compute lbp features
    """

    n_points = 8 * radius
    n_bins = n_points + 2
    lbp = local_binary_pattern(image, n_points, radius, 'uniform')
    lbp = lbp[mask[:,:,0] > 0]
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0.0, float(n_bins)))
    return hist


def compute_sim_features(image, mask):
    """
    similarity visual feature extraction
    """

    image_rgb32 = image.astype('float32')
    image_lab32 = color.rgb2lab(image).astype('float32')
    image_gray = color.rgb2gray(image)

    rgb_histogram = compute_histogram_color_combined(image_rgb32, mask)
    lab_histogram = compute_histogram_color_combined(image_lab32, mask, color='lab')
    lbp_features = compute_lbp_features(image_gray, mask)

    return np.hstack([rgb_histogram, lab_histogram, lbp_features])
    # return np.hstack([lbp_features])
