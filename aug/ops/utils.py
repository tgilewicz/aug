import cv2
import numpy as np
import random


def hsv_to_rgb(hsv):
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[0][0]

    return np.array([int(rgb[0]), int(rgb[1]), int(rgb[2])])


def random_bright_color():
    hsv = np.array([[[random.randint(0, 360),
                      random.randint(0, 50),
                      random.randint(100, 255)]]],
                   dtype=np.uint8)

    return hsv_to_rgb(hsv).astype(np.int32).tolist()


def random_dark_color():
    hsv = np.array([[[random.randint(0, 360),
                      random.randint(0, 200),
                      random.randint(30, 120)]]],
                   dtype=np.uint8)

    return hsv_to_rgb(hsv).astype(np.int32).tolist()


def fit_borders(image, horizontal_only=False, vertical_only=False):
    assert not (vertical_only and horizontal_only)
    tmp = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    tmp = 255 - tmp
    if np.sum(tmp) == 0:
        return image

    oy_sum = np.sum(tmp, axis=1)
    oy_not_zeros = np.where(oy_sum != 0)[0]

    ox_sum = np.sum(tmp, axis=0)
    ox_not_zeros = np.where(ox_sum != 0)[0]

    # display margins
    # tmp[oy_not_zeros[0], :] = 255
    # tmp[oy_not_zeros[-1], :] = 255
    # tmp[:, ox_not_zeros[0]] = 255
    # tmp[:, ox_not_zeros[-1]] = 255

    b = 2
    if not horizontal_only:
        image = image[max(oy_not_zeros[0] - b, 0):min(oy_not_zeros[-1] + b, image.shape[0] - 1), :]

    if not vertical_only:
        image = image[:, max(ox_not_zeros[0] - b, 0):min(ox_not_zeros[-1] + b, image.shape[1] - 1)]

    return image


def unify_num_of_channels(input_img, to_unify):
    if input_img.shape[2] == 4:
        to_unify = cv2.cvtColor(to_unify, cv2.COLOR_RGB2RGBA)
        to_unify[:, :, 3] = 255

    elif input_img.shape[2] == 1:
        to_unify = cv2.cvtColor(to_unify, cv2.COLOR_RGB2GRAY)

    return to_unify


class RingBuffer:

    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)

        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_sample(self):
        if self.start == 0 and self.end > 0:
            idx = random.randint(self.start, self.end - 1)
        elif self.start - self.end == 1:
            idx = random.randint(0, len(self.data) - 1)
        else:
            raise Exception('Unhandled index values')

        return self.data[idx]
