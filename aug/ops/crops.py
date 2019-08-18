import random
import numpy as np
import aug


@aug.perform_randomly
class Crop(aug.Operation):

    def __init__(self, shape):
        self._shape = shape
        self._pos = None

    def apply_on_image(self, image):
        assert image.shape[0] > self._shape[0] and image.shape[1] > self._shape[1], \
            "An image should be larger than a crop to be cut."

        if self._pos is None:
            x = random.randint(0, image.shape[1] - self._shape[1])
            y = random.randint(0, image.shape[0] - self._shape[0])
            self._pos = x, y
        else:
            x, y = self._pos

        return image[y:y+self._shape[0], x:x+self._shape[1]].copy()

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])


@aug.perform_randomly
class GridCrop(aug.Operation):

    def __init__(self, shape):
        """A crop shape should be a multiple of an image size, otherwise
            right and bottom margins will be ignored.
        """

        self._shape = shape
        self._pos = None

    def apply_on_image(self, image):
        assert image.shape[0] > self._shape[0] and image.shape[1] > self._shape[1], \
            "An image should be larger than a crop to be cut."

        if self._pos is None:
            x = random.choice(list(range(0, image.shape[0] - self._shape[0], self._shape[0])))
            y = random.choice(list(range(0, image.shape[1] - self._shape[1], self._shape[1])))

            self._pos = x, y
        else:
            x, y = self._pos

        return image[y:y+self._shape[0], x:x+self._shape[1]].copy()

    def apply_on_masks(self, masks):
        return np.array([self.apply_on_image(mask) for mask in list(masks)])

