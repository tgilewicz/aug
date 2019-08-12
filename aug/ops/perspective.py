import random

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from aug import Operation, perform_randomly, utils


@perform_randomly
class PerspectiveDistortion(Operation):

    def __init__(self, max_warp=0.2, input_mtx=None, return_mtx=False):
        self._max_warp = max_warp
        self._mtx = input_mtx
        self._return_mtx = return_mtx

    def get_mtx(self, im_height, im_width):
        b = int(min(im_height, im_width) * self._max_warp)
        r = random.randint

        pts2 = np.float32([[0, 0], [im_width - 1, 0], [0, im_height - 1],
                           [im_width - 1, im_height - 1]])

        pts1 = np.float32([[r(0, b), r(0, b)], [im_width - 1 - r(0, b),
                                                r(0, b)], [r(0, b), im_height - 1 - r(0, b)],
                           [im_width - 1 - r(0, b), im_height - 1 - r(0, b)]])

        return cv2.getPerspectiveTransform(pts1, pts2)

    def transform_perspective_and_get_matrix(self, img):
        """
            Find four random points within image and apply perspective transformation
        Args:
            img: input image
            max_warp: limiter of points positions
            mtx: perspective matrix
        """
        im_height, im_width = img.shape[:2]

        if self._mtx is None:
            self._mtx = self.get_mtx(im_height, im_width)

        return cv2.warpPerspective(img, self._mtx, (im_width, im_height)), self._mtx

    def apply_on_image(self, img):
        image, mtx = self.transform_perspective_and_get_matrix(img)

        if self._return_mtx:
            return image, mtx

        return image

    def apply_on_annotations(self, annotations):
        """Apply transformation on set of points. """

        if self._mtx is not None:
            annotations = annotations.astype(np.float32)
            annotations = cv2.perspectiveTransform(annotations, self._mtx)
            annotations = annotations.astype(np.int32)

        return annotations

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class ElasticDistortion(Operation):
    """
        Based on: https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py

    """

    def __init__(self,
                 alpha=100.,
                 sigma=10.,
                 alpha_affine_range=10.,
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        self._alpha = alpha
        self._sigma = sigma
        self._alpha_affine = alpha_affine_range
        self._interpolation = interpolation
        self._border_mode = border_mode

        self._alpha = float(self._alpha)
        self._sigma = float(self._sigma)
        self._alpha_affine = float(self._alpha_affine)

        self._mapx = None
        self._mapy = None
        self._matrix = None

    def apply_on_image(self, image):
        h, w = image.shape[:2]

        if self._mapx is not None and self._mapy is not None and self._matrix is not None:
            image = cv2.warpAffine(image,
                                   self._matrix, (w, h),
                                   flags=self._interpolation,
                                   borderMode=self._border_mode)

            return cv2.remap(image, self._mapx, self._mapy, self._interpolation, borderMode=self._border_mode)

        # If method is called first time:
        center_square = np.float32((h, w)) // 2     # Random affine
        square_size = min((h, w)) // 3

        pts1 = np.float32([
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size
        ])
        pts2 = pts1 + np.random.uniform(
            -self._alpha_affine, self._alpha_affine, size=pts1.shape).astype(np.float32)
        self._matrix = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image,
                               self._matrix, (w, h),
                               flags=self._interpolation,
                               borderMode=self._border_mode)

        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self._sigma)
        dx = np.float32(dx * self._alpha)

        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self._sigma)
        dy = np.float32(dy * self._alpha)

        x, y = np.meshgrid(np.arange(w), np.arange(h))

        self._mapx = np.float32(x + dx)
        self._mapy = np.float32(y + dy)

        return cv2.remap(image, self._mapx, self._mapy, self._interpolation, borderMode=self._border_mode)

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class GridDistortion(Operation):
    """
        Based on: https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py

    """

    def __init__(self,
                 num_steps=(10, 10),
                 distort_limit=(.1, 2.),
                 interpolation=cv2.INTER_LINEAR,
                 maintain_size=True):
        self._num_steps = num_steps
        self._xsteps = [
            1 + random.uniform(distort_limit[0], distort_limit[1]) for _ in range(num_steps[0] + 1)
        ]
        self._ysteps = [
            1 + random.uniform(distort_limit[0], distort_limit[1]) for _ in range(num_steps[1] + 1)
        ]
        self._interpolation = interpolation
        self._maintain_size = maintain_size

    def apply_on_image(self, img):
        h, w = img.shape[:2]

        x_step = w // self._num_steps[0]
        xx = np.zeros(w, np.float32)
        prev = 0
        for idx, x in enumerate(range(0, w, x_step)):
            start = x
            end = x + x_step
            if end > w:
                end = w
                cur = w
            else:
                cur = prev + x_step * self._xsteps[idx]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        y_step = h // self._num_steps[1]
        yy = np.zeros(h, np.float32)
        prev = 0
        for idx, y in enumerate(range(0, h, y_step)):
            start = y
            end = y + y_step
            if end > h:
                end = h
                cur = h
            else:
                cur = prev + y_step * self._ysteps[idx]

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        img = cv2.remap(img,
                        map_x,
                        map_y,
                        interpolation=self._interpolation,
                        borderMode=cv2.BORDER_CONSTANT)

        img = 255 - utils.fit_borders(255 - img)

        if self._maintain_size:
            img = cv2.resize(img, (w, h))

        return img


@perform_randomly
class OpticalDistortion(Operation):
    """
        Based on: https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py

    """

    def __init__(self,
                 distort_limit_x=(-.003, .003),
                 distort_limit_y=(-.003, .003),
                 shift_limit=(-.1, .1),
                 interpolation=cv2.INTER_LINEAR,
                 border_color=(0, 0, 0)):
        self._shift_limit = shift_limit
        self._interpolation = interpolation
        self._border_color = border_color

        self._k_x = random.uniform(*distort_limit_x)
        self._k_y = random.uniform(*distort_limit_y)
        self._dx = random.uniform(*shift_limit)
        self._dy = random.uniform(*shift_limit)

    def apply_on_image(self, img):
        h, w = img.shape[:2]

        dx = round(w * self._dx)
        dy = round(h * self._dy)
        k_x = self._k_x * w
        k_y = self._k_y * h
        fx = w
        fy = w

        cx = w * 0.5 + dx
        cy = h * 0.5 + dy

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        distortion = np.array([k_x, k_y, 0, 0, 0], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (w, h),
                                                 cv2.CV_32FC1)

        img = cv2.remap(img,
                        map1,
                        map2,
                        interpolation=self._interpolation,
                        borderMode=0,
                        borderValue=self._border_color)

        img[:, :, :3] -= np.array(self._border_color).astype(np.uint8)
        img = 255 - utils.fit_borders(255 - img)
        img[:, :, :3] += np.array(self._border_color).astype(np.uint8)

        return cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
