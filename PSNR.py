import numpy as np
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 9999999
    PIXEL_MAX = 2.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))