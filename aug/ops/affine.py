import random

import cv2
import numpy as np

import aug
from aug import Operation, perform_randomly


class RotationWithBound(Operation):

    def __init__(self,
                 angle=aug.uniform(-30, 30),
                 interpolation=cv2.INTER_LINEAR,
                 mode='replicate',
                 change_size=True):
        self._angle = angle
        self._interpolation = interpolation
        self._mode = mode
        self._change_size = change_size
        self.mtx = None
        self.new_width = None
        self.new_height = None

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]

        if self._mode == "zeros":
            mode = cv2.BORDER_CONSTANT
        elif self._mode == "replicate":
            mode = cv2.BORDER_REPLICATE
        else:
            raise Exception("Unknown mode value.")

        if self._change_size:
            center_x, center_y = im_width // 2, im_height // 2

            if self.mtx is None:
                self.mtx = cv2.getRotationMatrix2D((center_x, center_y), self._angle, 1.0)
                cos = np.abs(self.mtx[0, 0])
                sin = np.abs(self.mtx[0, 1])

                self.new_width = int((im_height * sin) + (im_width * cos))
                self.new_height = int((im_height * cos) + (im_width * sin))

                # Adjust rotation matrix to take translation into account
                self.mtx[0, 2] += (self.new_width / 2) - center_x
                self.mtx[1, 2] += (self.new_height / 2) - center_y

            return cv2.warpAffine(image,
                                  self.mtx, (self.new_width, self.new_height),
                                  flags=self._interpolation,
                                  borderMode=mode)

        if self.mtx is None:
            self.mtx = cv2.getRotationMatrix2D((im_width / 2, im_height / 2), self._angle, 1.0)

        img = cv2.warpAffine(image,
                             self.mtx, (im_width, im_height),
                             flags=self._interpolation,
                             borderMode=mode)

        return img

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class Rotation(Operation):

    def __init__(self, angle=30, mode='replicate'):
        self._angle = angle
        self._mode = mode
        self._mtx = None

        self._w_ratio = 1.
        self._h_ratio = 1.

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]

        rotation_with_bound = RotationWithBound(self._angle,
                                                mode=self._mode)

        image = rotation_with_bound.apply_on_image(image)
        self._mtx = rotation_with_bound.mtx

        tmp_h, tmp_w = image.shape[:2]
        self._w_ratio = tmp_w / im_width
        self._h_ratio = tmp_h / im_height

        return cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_CUBIC)

    def apply_on_annotations(self, annotations):
        if self._mtx is None:
            return annotations

        polygons = []
        for polygon in annotations:
            points = []
            for point in polygon:
                point = np.array([point[0], point[1], 1])
                rotated = np.dot(self._mtx, point.T).astype(int).tolist()
                points.append(rotated)
            polygons.append(points)
        rotated = np.array(polygons).astype(np.float32)
        rotated[:, :, 0] /= self._w_ratio
        rotated[:, :, 1] /= self._h_ratio

        return rotated

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class Stretch(Operation):

    def __init__(self, x_scale=0.5, y_scale=0.5):
        assert x_scale > 0 and y_scale > 0
        self._x_scale = x_scale
        self._y_scale = y_scale

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]

        if random.getrandbits(1):
            new_w = int(self._x_scale * im_width)
            new_w, new_h = int(new_w), im_height
        else:
            new_h = int(self._y_scale * im_height)
            new_w, new_h = im_width, int(new_h)

        return cv2.resize(image, (new_w, new_h))


@perform_randomly
class Rotation90(Operation):

    def __init__(self, iterations=None):
        self._iterations = iterations if iterations is not None else random.randint(0, 3)

    def apply_on_image(self, image):
        for _ in range(self._iterations):
            image = np.ascontiguousarray(np.rot90(image))
        return image

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class VerticalFlip(Operation):

    def __init__(self):
        self._h = None

    def apply_on_image(self, img):
        self._h = img.shape[0]
        return np.ascontiguousarray(img[::-1, ...])

    def apply_on_annotations(self, annotations):
        if self._h is not None:
            annotations[:, :, 1] = self._h - annotations[:, :, 1]
            annotations = aug.AnnotationOrderFix().apply_on_annotations(annotations)

        return annotations

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class HorizontalFlip(Operation):

    def __init__(self):
        self._w = None

    def apply_on_image(self, img):
        self._w = img.shape[1]
        return np.ascontiguousarray(img[:, ::-1, ...])

    def apply_on_annotations(self, annotations):
        if self._w is not None:
            annotations[:, :, 0] = self._w - annotations[:, :, 0]
            annotations = aug.AnnotationOrderFix().apply_on_annotations(annotations)

        return annotations

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@perform_randomly
class Transposition(Operation):

    def apply_on_image(self, img):
        return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


@perform_randomly
class Zoom(Operation):

    def __init__(self, margin=0.1):
        assert 0. < margin < .5

        self._margin = margin
        self._top = None
        self._left = None
        self._w_ratio = 1.
        self._h_ratio = 1.
        self._shape = None

    def apply_on_image(self, image):
        self._shape = h, w = image.shape[:2]
        h_abs_margin, w_abs_margin = int(h * self._margin), int(w * self._margin)

        if self._left is None:
            self._left = int(w * random.uniform(0, self._margin))
        right = w_abs_margin - self._left

        if self._top is None:
            self._top = int(h * random.uniform(0, self._margin))
        bottom = h_abs_margin - self._top

        image = image[self._top:h - bottom, self._left:w - right]

        tmp_h, tmp_w = image.shape[:2]
        self._w_ratio = tmp_w / w
        self._h_ratio = tmp_h / h

        return cv2.resize(image, (w, h))

    def apply_on_annotations(self, annotations):
        if self._left is not None and self._top is not None:
            annotations = annotations.astype(np.float32)
            annotations[:, :, 0] -= self._left
            annotations[:, :, 1] -= self._top
            annotations[:, :, 0] /= self._w_ratio
            annotations[:, :, 1] /= self._h_ratio

        return annotations

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


class ConstantAspectRatioScaling(Operation):
    """ Resize image with constant aspect ratio. """

    def __init__(self, dst_shape, interpolation=None):
        self._dst_shape = dst_shape
        self._interpolation = interpolation
        self.scale_ratio = None

    def apply_on_image(self, image):
        height, width = self._dst_shape
        h, w = image.shape[:2]

        assert width is not None and height is not None
        assert h is not None and w is not None

        if self._interpolation is None:
            self._interpolation = cv2.INTER_AREA if \
                h > self._dst_shape[0] or w > self._dst_shape[1] else cv2.INTER_LINEAR

        if w < h:
            r = height / float(h)
            size = (int(round(w * r)), height)

        else:
            r = width / float(w)
            size = (width, int(round(h * r)))

        self.scale_ratio = r
        return cv2.resize(image, size, interpolation=self._interpolation)

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])

    def apply_on_annotations(self, annotations):
        # TODO test
        if self.scale_ratio is not None:
            annotations[:, :, :] *= self.scale_ratio

        return annotations
