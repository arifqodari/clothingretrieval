import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

if __name__ == "__main__":
    # m = io.imread(sys.argv[1])
    m = np.loadtxt(sys.argv[1], dtype='int')
    print m.dtype

    mask = np.zeros(m.shape, dtype='uint8')
    mask[m >= 1] = 255

    fig = plt.figure("vis")
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("original image")
    ax1.imshow(mask)
    plt.show()
