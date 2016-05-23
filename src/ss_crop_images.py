import os
import numpy as np
import cv2
import argparse


YMID_RANGE = (298, 303)


def crop_method1(image):
    """
    first method
    crop the main image
    """

    # convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)

    # compute edges
    edges = cv2.Canny(image_gray, 50, 100)
    edges[edges > 128] = 255
    edges[edges < 255] = 0

    # find the middle horizontal line
    line = edges[YMID_RANGE[0]:YMID_RANGE[1],:]
    line_length = line.sum(axis=1)
    rows = np.where(line_length == line_length.max())[0]

    # compare the length of left and right lines
    if line[rows, 0:300].sum() > line[rows, 300:600].sum():
        return image[:, 300:600]
    else:
        return image[:, 0:300]


def crop_method2(image):
    """
    second method
    crop the largest object appearance
    """

    # convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)

    # compute edges
    edges = cv2.Canny(image_gray, 50, 100)
    edges[edges > 128] = 255
    edges[edges < 255] = 0

    # find the longest horizontal and vertical lines
    hline_length = edges.sum(axis=1) / 255
    vline_length = edges.sum(axis=0) / 255
    hline_y = min(np.where(hline_length == hline_length.max())[0]) if hline_length.max() > 0.75 * image.shape[1] else image.shape[0]
    vline_x = min(np.where(vline_length == vline_length.max())[0]) if vline_length.max() > 0.75 * image.shape[0] else image.shape[1]

    # crop the image based on detected line
    cropped_image = image
    if vline_length.max() > 0.75 * image.shape[0] or hline_length.max() > 0.75 * image.shape[1]:
        y_range = (0, hline_y) if hline_y > image.shape[0] - hline_y else (hline_y, image.shape[0])
        x_range = (0, vline_x) if vline_x > image.shape[1] - vline_x else (vline_x, image.shape[1])
        cropped_image = image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    else:
        cropped_image = crop_method1(image)

    # get the object area
    image_gray = cv2.cvtColor(cropped_image, cv2.cv.CV_BGR2GRAY)
    edges = cv2.Laplacian(image_gray, -1, ksize=5)
    edges[edges > 128] = 255
    edges[edges < 255] = 0
    object_area = np.zeros(edges.shape, dtype='uint8')
    object_area = object_area | edges
    object_area[image_gray < 190] = 255

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

    return cropped_image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


def crop(image, cat=None):
    special_cats = ('bag', 'shoes', 'accessory')

    if cat in special_cats:
        return crop_method2(image)
    else:
        return crop_method1(image)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-dir', help='input directory')
    parser.add_argument('--out-dir', help='output directory')
    parser.add_argument('--cat', help='category')
    args = parser.parse_args()
    return args.inp_dir, args.out_dir, args.cat


def main():
    input_dir, output_dir, cat = parse_arguments()
    filenames = os.listdir(input_dir)

    for i, filename in enumerate(filenames):
        print "%i from %i" % (i+1, len(filenames))
        image_path = os.path.join(input_dir, filename)

        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            if image.shape != (600, 600, 3): continue

            cropped_image = crop(image, cat)
            cv2.imwrite(os.path.join(output_dir, filename), cropped_image)


if __name__ == "__main__":
    main()
