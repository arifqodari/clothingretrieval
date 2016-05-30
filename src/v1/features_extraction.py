"""
Feature Extracion
"""


import cv2
import numpy as np
import serialization as srz
from skimage import color as skcolor
from skimage.feature import local_binary_pattern as sklbp


class SimilarityFeaturesExtractor:
    """
    Feature extractor for similarity search
    The input image should be RGB uint8

    Similarity features are:
    1) Histogram of RGB
    2) Histogram of Lab
    3) LBP uniform features
    """

    def __init__(self, hist_nbins=8, lbp_radius=2):
        self.hist_nbins = hist_nbins
        self.lbp_radius = lbp_radius
        self.lbp_npoints = lbp_radius * 8
        self.lbp_nbins = self.lbp_npoints + 2
        self.lbp_method = 'uniform'

    ###########################################################################
    # Public methods
    ###########################################################################

    def extract(self, image, mask=None):
        if mask is None:
            mask = np.ones(image.shape[:2], dtype='uint8')  # default mask

        lab = skcolor.rgb2lab(image).astype('float32')  # float32
        gray = skcolor.rgb2gray(image)  # float64

        rgb_hist = self.__compute_color_histogram(image, mask)  # uint8
        lab_hist = self.__compute_color_histogram(lab, mask, color='lab')
        lbp = self.__compute_lbp(gray, mask)

        return np.hstack([rgb_hist, lab_hist, lbp])

    def extracts(self, images, masks=None):
        features_set = []
        for i in xrange(0, len(images)):
            if masks is not None:
                features = self.extract(images[i], masks[i])
            else:
                features = self.extract(images[i], None)
            features_set.append(features)

        return np.array(features_set)

    ###########################################################################
    # Private methods
    ###########################################################################

    def __compute_color_histogram(self, image, mask, color='rgb'):
        """
        3D color histogram
        """

        if color == 'rgb':
            ranges = [0, 255, 0, 255, 0, 255]
        elif color == 'lab':
            ranges = [0.0, 100.0, -127.0, 127.0, -127.0, 127.0]

        num_bins = [self.hist_nbins, self.hist_nbins, self.hist_nbins]
        hist = cv2.calcHist([image], [0, 1, 2], mask, num_bins, ranges)

        return cv2.normalize(hist, norm_type=cv2.NORM_MINMAX).flatten()

    def __compute_lbp(self, image, mask):
        """
        Histogram of local binary pattern features
        """

        lbp = sklbp(image, self.lbp_npoints, self.lbp_radius, self.lbp_method)
        lbp = lbp[mask > 0]
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=self.lbp_nbins,
                               range=(0.0, float(self.lbp_nbins)))
        return hist
