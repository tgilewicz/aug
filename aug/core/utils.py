from PIL import Image
import numpy as np
import random
import cv2
import os


def weighted_choice(choices_dict, probability_sum=None):
    if probability_sum is None:
        probability_sum = sum(v for k, v in choices_dict.items())

    r = random.uniform(0, probability_sum)
    upto = 0
    for k, v in choices_dict.items():
        if upto + v >= r:
            return k
        upto += v
    return ""


def show(img, title='Image'):
    if isinstance(img, Image.Image):
        im = np.array(img)
        img = im.reshape((im.shape[0], im.shape[1], -1))

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_image(file_name):
    directory = os.path.abspath(__file__)

    for _ in range(2):
        directory = os.path.dirname(directory)

    src_dir = os.path.join(directory, 'images', file_name)
    if os.path.exists(src_dir):
        return cv2.imread(src_dir)
    else:
        raise Exception("Image file not found. (\"{}\")".format(src_dir))
