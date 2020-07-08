from __future__ import print_function

import torch

torch.backends.cudnn.bencmark = True
import cv2
import numpy as np
from .matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],

               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]

    crop_size = (112, 112)

    src_pts = np.array(src_pts).reshape(2, 5)
    src_pts = np.transpose(src_pts)

    s = np.array(src_pts).astype(np.float32)

    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)

    face_img = cv2.warpAffine(src_img, tfm, crop_size)

    return face_img


# landmark = [138, 193, 217, 185, 170, 234, 151, 280, 216, 275]
# img = cv2.imread('images\\zyf.jpg')
# cv2_alignment = alignment(img, landmark)
# cv2.imwrite("alignment.jpg", cv2_alignment)
# cv2.imshow("img", cv2_alignment)
# cv2.waitKey(0)
