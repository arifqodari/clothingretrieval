import os
import argparse
import numpy as np
import cv2

from skimage import io
from skimage import transform as tran
from skimage import util as skutil

import features as feat


def crop_person(anno_dir, photo_dir, output_photo_dir, output_anno_dir, size=None):
    """
    features extraction to train unary potential
    """

    # get all csv annotation files
    csv_dirs = os.listdir(anno_dir)

    for i, csv_file in enumerate(csv_dirs):
        print "%i from %i" % (i, len(csv_dirs))

        csv_path = os.path.join(anno_dir, csv_file)
        image_path = os.path.join(photo_dir, csv_file.split(".")[0] + ".jpg")

        if os.path.isfile(csv_path):

            # load image and label
            image = io.imread(image_path)
            pixel_class = io.imread(csv_path)

            # create mask
            mask = np.zeros(pixel_class.shape, dtype='uint8')
            mask[pixel_class > 0] = 255

            # get contours
            edges = cv2.Canny(mask, 50, 100)
            contours, _ = cv2.findContours(edges, mode=cv2.cv.CV_RETR_TREE, method=cv2.cv.CV_CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            # get bounding rectangle
            x,y,w,h = cv2.boundingRect(np.vstack(contours))

            # crop the image and label
            im_out = image[y:y+h, x:x+w, :]
            pixel_out = pixel_class[y:y+h, x:x+w]

            # resize or pad if necessary
            if size is not None:

                # resize
                im_shape = list(im_out.shape[0:2])
                if im_shape[0] > size[0] or im_shape[1] > size[1]:
                    diff = [size[0] / float(im_shape[0]), size[1] / float(im_shape[1])]
                    scale = size[diff.index(min(diff))] / float(im_shape[diff.index(min(diff))])
                    im_out = skutil.img_as_ubyte(tran.rescale(im_out, scale=(scale, scale)))
                    pixel_out = tran.rescale(pixel_out, scale=(scale, scale), preserve_range=True).astype('uint8')

                # pad
                pad_h = size[0] - im_out.shape[0]
                pad_w = size[1] - im_out.shape[1]
                im_out = np.pad(im_out, pad_width=((pad_h / 2, pad_h - (pad_h / 2)), (pad_w / 2, pad_w - (pad_w / 2)), (0, 0)), mode = 'edge')
                pixel_out = np.pad(pixel_out, pad_width=((pad_h / 2, pad_h - (pad_h / 2)), (pad_w / 2, pad_w - (pad_w / 2))), mode = 'constant', constant_values = 0)


            # save the image and label
            new_image_path = os.path.join(output_photo_dir, csv_file.split(".")[0] + ".jpg")
            new_csv_path = os.path.join(output_anno_dir, csv_file.split(".")[0] + ".png")
            io.imsave(new_image_path, im_out)
            io.imsave(new_csv_path, pixel_out)


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-dir', help='csv_parts annotation directory')
    parser.add_argument('--photo-dir', help='photo directory')
    parser.add_argument('--output-photo-dir', help='output photo directory')
    parser.add_argument('--output-anno-dir', help='output annotation directory')
    parser.add_argument('--max-size', help='max size of new image')
    args = parser.parse_args()

    # perform cropping
    max_size = list(np.array(args.max_size.split(',')).astype('int')) if args.max_size != '-1' else None
    crop_person(args.anno_dir, args.photo_dir, args.output_photo_dir, args.output_anno_dir, max_size)
