import numpy as np
import os


PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
DEFAULT_LABEL_DICT_FILE = os.path.join(PROJECT_DIR, 'data/processed/categories/ccp_to_ss_cats.csv')


def get_label_dictionary(label_dict_file=DEFAULT_LABEL_DICT_FILE, fileformat='int'):
    """
    return label dictionary information
    """

    if label_dict_file is None:
        label_dict_file = DEFAULT_LABEL_DICT_FILE

    if fileformat is None:
        label_dict = np.loadtxt(label_dict_file, delimiter=',', dtype=fileformat)
    else:
        label_dict = np.loadtxt(label_dict_file, delimiter=',', dtype=fileformat)

    return dict(label_dict)


def get_image_paths(dir_path):
    """
    from given directory path, return a set of image paths
    """
    image_dir = os.listdir(dir_path)
    image_paths = []

    for i, image_file in enumerate(image_dir):
        if len(image_file) == 23:
            image_path = {}
            [filename, extension] = image_file.split(".")

            # rgb image path
            image_path['full'] = os.path.join(dir_path, image_file)

            # mask image path
            image_mask_file = filename + "_mask." + extension
            image_path['mask'] = os.path.join(dir_path, image_mask_file)

            # crop image
            image_crop_file = filename + "_crop." + extension
            image_path['crop'] = os.path.join(dir_path, image_crop_file)

            image_paths.append(image_path)

    return image_paths


