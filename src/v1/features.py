"""
Features
"""


import numpy as np
import serialization as srz
from sklearn import neighbors as skneigh


class SimilarityFeaturesCollection:
    """
    Similarity features
    consist of:
    - features_set
    - product ids
    """

    def __init__(self, features_set=None, product_ids=None, category=None):
        self.features_set = [] if features_set is None else features_set
        self.product_ids = [] if product_ids is None else product_ids
        self.category = category

    ###########################################################################
    # Public methods
    ###########################################################################

    def init_kd_tree(self):
        self.kd_tree = skneigh.KDTree(np.array(self.features_set))

    def search(self, features, k=10):
        """
        KD-Tree based similarity search
        make sure the tree is already initialized
        """

        if self.kd_tree is None:
            self.init_kd_tree()

        dists, indices = self.kd_tree.query(features.reshape(1, -1), k=k)
        return [self.product_ids[idx] for idx in indices[0]]

    ###########################################################################
    # Setter and Getter
    ###########################################################################

    def reload(self, features_set, product_ids):
        self.features_set, self.product_ids = features_set, product_ids

    def append(self, features, product_id):
        self.features_set.append(features)
        self.product_ids.append(product_id)

    def get_features_set(self):
        return self.features_set

    def get_product_ids(self):
        return self.product_ids

    def get_category(self):
        return self.category

    ###########################################################################
    # Serialization
    ###########################################################################

    def to_string(self):
        return srz.obj_to_str((self.features_set, self.product_ids))

    def reload_from_string(self, string):
        self.features_set, self.product_ids = srz.str_to_obj(string)
