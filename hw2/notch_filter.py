import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
def IdealNotchFilter(fshiftinput, points, d0):
    m = fshiftinput.shape[0]
    n = fshiftinput.shape[1]
    for u in range(m):
        for v in range(n):
            for d in range(len(points)):
                u0 = points[d][0]
                v0 = points[d][1]
                u0, v0 = v0, u0
                d1 = pow(pow(u - u0, 2) + pow(v - v0, 2), 1)
                d2 = pow(pow(u + u0, 2) + pow(v + v0, 2), 1)
                if d1 <= d0 or d2 <= d0:
                    fshiftinput[u][v] *= 0.0
    f_ishift = np.fft.ifftshift(fshiftinput)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back
