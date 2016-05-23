import argparse
import dlib
import cv2

from skimage import io
from skimage import color

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filename', help='output directory')
    parser.add_argument('--image-path', help='output directory')
    args = parser.parse_args()

    # load detector
    detector = dlib.simple_object_detector(args.model_filename)

    # load image
    image = io.imread(args.image_path)
    image = cv2.GaussianBlur(image, (5,5), 0)

    # perform prediction
    rects = detector(image)

    # visualize result
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(image)
    win.add_overlay(rects)
    dlib.hit_enter_to_continue()
