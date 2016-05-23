import pandas as pd
import numpy as np
import argparse
import urllib2
import os


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='input csv file')
    parser.add_argument('--out-dir', help='output directory')
    args = parser.parse_args()

    data = pd.read_csv(args.csv)

    current_image_id = ''
    ii = -1
    for index, row in data.iterrows():
        if 'id' in row.keys():
            image_id = str(row['id'])
        else:
            image_id = row['url'].split('/')[-1].split('_')[0]

            if image_id == current_image_id:
                ii += 1
            else:
                ii = 0

            current_image_id = image_id
            image_id += '_' +  str(ii)

        filename = os.path.join(args.out_dir, image_id + '.' + row['url'].split('.')[-1])

        # check file exist
        if os.path.isfile(filename):
            continue

        print filename

        try:
            f = urllib2.urlopen(row['url'])
            imagedata = f.read()
            with open(filename, "wb") as im:
                im.write(imagedata)
            f.close()
        except urllib2.HTTPError, e:
            print e.code
            continue
        except urllib2.URLError, e:
            print e.args
            continue
