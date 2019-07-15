import random

import cv2
import numpy as np
from aug import perform_randomly, Operation
import aug


@perform_randomly
class MedianBlur(Operation):

    def __init__(self, ksize_norm=.05):
        self._ksize_norm = ksize_norm

    def apply_on_image(self, image):
        k_size = int(min(image.shape[:2]) * self._ksize_norm)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size

        if k_size <= 2:
            return image

        return cv2.medianBlur(image, k_size)


@perform_randomly
class VariableBlur(Operation):

    def __init__(self, modes=('linear', 'radial'), ksize_norm=.2):
        self._ksize_norm = ksize_norm
        self._modes = modes
        for elem in modes:
            assert elem in ['linear', 'radial']

    def radial_mask(self, image):
        max_dim = max(image.shape)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = aug.RadialGradient(inner_color=255,
                                  outer_color=10,
                                  max_distance=max_dim *
                                  random.uniform(.35, .65)).apply_on_image(mask)

        return mask

    def linear_mask(self, image):
        mask = np.zeros(image.shape, dtype=np.uint8)

        edge1 = random.uniform(0.7, 1.)
        edge2 = random.uniform(0, .3)

        vertical = aug.LinearGradient(p=1.0, edge_brightness=(edge1, edge2))
        horizontal = aug.LinearGradient(p=1.0, edge_brightness=(edge1, edge2))

        op = vertical if random.getrandbits(1) else horizontal
        mask = op.apply_on_image(mask)
        return mask

    def apply_on_image(self, image):
        k_size = int(min(image.shape[:2]) * self._ksize_norm)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        if k_size <= 2:
            return image

        image_blurred = cv2.blur(image.copy(), ksize=(k_size, k_size))

        mode = random.choice(self._modes)

        if 'linear' == mode:
            mask = self.linear_mask(image)
        elif 'radial' == mode:
            mask = self.radial_mask(image)

        image = image.astype(float)
        image_blurred = image_blurred.astype(float)
        mask = mask.astype(float) / 255

        image = cv2.multiply(mask, image).astype(np.uint8)
        image_blurred = cv2.multiply(1.0 - mask, image_blurred).astype(np.uint8)

        return cv2.add(image, image_blurred)


@perform_randomly
class MotionBlur(Operation):

    def __init__(self, ksize_norm=.1):
        self._ksize_norm = ksize_norm

    def apply_on_image(self, img):
        k_size = int(min(img.shape[:2]) * self._ksize_norm)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        if k_size <= 2:
            return img

        x1, x2 = random.randint(0, k_size - 1), random.randint(0, k_size - 1)
        y1, y2 = random.randint(0, k_size - 1), random.randint(0, k_size - 1)

        kernel_mtx = np.zeros((k_size, k_size), dtype=np.uint8)
        cv2.line(kernel_mtx, (x1, y1), (x2, y2), 1, thickness=1)
        return cv2.filter2D(img, -1, kernel_mtx / np.sum(kernel_mtx))


@perform_randomly
class GaussianBlur(Operation):

    def __init__(self, ksize_norm=.4, sigma=5, direction=None):
        self.ksize_norm = ksize_norm
        self._sigma = sigma
        self._direction = direction

        assert direction in ('horizontal', 'vertical', None)

    def apply_on_image(self, image):
        k_size = int(min(image.shape[:2]) * self.ksize_norm)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        if k_size <= 2:
            return image

        if self._direction == "horizontal":
            return cv2.GaussianBlur(image, (k_size, 1), sigmaX=self._sigma, sigmaY=self._sigma)
        elif self._direction == "vertical":
            return cv2.GaussianBlur(image, (1, k_size), sigmaX=self._sigma, sigmaY=self._sigma)

        return cv2.GaussianBlur(image, (k_size, k_size), sigmaX=self._sigma, sigmaY=self._sigma)
