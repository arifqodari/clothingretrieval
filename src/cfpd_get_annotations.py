import argparse
import os
import numpy as np


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-dir', help='raw annotation directory')
    parser.add_argument('--out-dir', help='output annotation directory')
    args = parser.parse_args()

    # get list of filenames
    filenames = os.listdir(args.inp_dir)

    for i, csv_filename in enumerate(filenames):

        # only process superpixel csv file
        if csv_filename[-7:-4] == 'seg':

            print csv_filename

            sps_filename = os.path.join(args.inp_dir, csv_filename)
            cat_filename = os.path.join(args.inp_dir, csv_filename[:-8] + '.cat.csv')

            # load superpixels and category information
            sps = np.loadtxt(sps_filename, delimiter=',', dtype='int')
            cat = np.loadtxt(cat_filename, delimiter=',', dtype='int')

            annotation = np.zeros(sps.shape, dtype='int')

            for i, segment in enumerate(np.unique(sps)):

                # since the original file is written in matlab
                # the category_id starts from 1
                # we should convert to zero-based id
                annotation[sps == segment] = cat[segment] - 1

            # write the annotation file
            out_file = os.path.join(args.out_dir, csv_filename[:-12] + '.csv')
            np.savetxt(out_file, annotation, fmt='%i', delimiter=',')
