import numpy as np
import aug


class Pass(aug.Operation):
    """ Pass an unmodified image."""

    def __init__(self):
        super().__init__()

    def apply_on_image(self, image):
        return image


class Pad(aug.Operation):
    """ Pad image to fixed size. """

    def __init__(self, shape, horizontal="center", vertical="center", value=0):
        assert horizontal in ['left', 'center', 'right']
        assert vertical in ['top', 'center', 'bottom']

        self._horizontal = horizontal
        self._vertical = vertical
        self._value = value
        self._shape = shape
        self._left = None
        self._top = None

    def apply_on_image(self, image):
        diff_h = self._shape[0] - image.shape[0]
        diff_w = self._shape[1] - image.shape[1]

        if self._horizontal == "center":
            self._left = diff_w // 2
            right = diff_w - self._left
        elif self._horizontal == "left":
            self._left = 0
            right = diff_w
        elif self._horizontal == "right":
            self._left = diff_w
            right = 0
        else:
            raise Exception("Unknown value")

        if self._vertical == "center":
            self._top = diff_h // 2
            bottom = diff_h - self._top
        elif self._vertical == "top":
            self._top = 0
            bottom = diff_h
        elif self._vertical == "bottom":
            self._top = diff_h
            bottom = 0
        else:
            raise Exception("Unknown value")

        image = np.pad(image, ((self._top, bottom), (self._left, right), (0, 0)),
                       'constant', constant_values=(self._value, self._value))

        return image

    def apply_on_annotations(self, annotations):
        if self._left is not None and self._top is not None:
            annotations[:, :, 0] += self._left
            annotations[:, :, 1] += self._top
        return annotations

    def apply_on_masks(self, masks):
        # TODO handle values other than 0
        return np.array([self.apply_on_image(mask) for mask in list(masks)])
