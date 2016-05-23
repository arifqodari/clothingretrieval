import numpy as np
import argparse


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='output file')
    parser.add_argument('--inp', metavar='inp', nargs='*', help='input files')
    args = parser.parse_args()

    full_arr = np.array([])

    # concat all csvs into one big csv file
    for inp in args.inp:
        print inp
        arr = np.load(inp)
        print arr.shape
        full_arr = np.append(full_arr, arr, axis=0) if full_arr.size else arr
        print full_arr.shape

    # save the final csv
    if full_arr.size:
        np.save(args.out, full_arr)
