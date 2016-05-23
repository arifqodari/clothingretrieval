import os
import numpy as np
import cv2
import argparse


YMID_RANGE = (298, 303)


def cld(image):
    # get the object area
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)
    edges = cv2.Canny(image_gray, 50, 100)
    edges[edges > 128] = 255
    edges[edges < 255] = 0
    object_area = np.zeros(edges.shape, dtype='uint8')
    object_area = object_area | edges
    object_area[image_gray < 190] = 255
    fg_area = object_area.copy()
    retval, rect = cv2.floodFill(fg_area, None, (0, 0), 255)
    fg_area = ~fg_area
    object_area = object_area | fg_area

    return object_area

    # get the largest object
    edges = cv2.Laplacian(object_area, -1, ksize=5)
    contours, _ = cv2.findContours(edges, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)
    max_rect_size = 0
    for i, contour in enumerate(contours):
        curve = cv2.approxPolyDP(contours[i], 20, closed=True)
        x,y,w,h = cv2.boundingRect(curve)
        if w * h > max_rect_size:
            max_rect_size = w * h
            rect = (x,y,w,h)

    return image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-dir', help='input directory')
    parser.add_argument('--out-dir', help='output directory')
    args = parser.parse_args()
    return args.inp_dir, args.out_dir


def main():
    input_dir, output_dir = parse_arguments()
    filenames = os.listdir(input_dir)

    for i, filename in enumerate(filenames):
        print "%i from %i" % (i+1, len(filenames))
        image_path = os.path.join(input_dir, filename)

        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            if image.shape != (600, 600, 3): continue

            cropped_image = cld(image)
            cv2.imwrite(os.path.join(output_dir, filename), cropped_image)


if __name__ == "__main__":
    main()
