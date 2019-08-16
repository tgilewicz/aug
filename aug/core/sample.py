import numpy as np
from .utils import read_image


class Sample(object):
    """Class representing a single training sample. """

    def __init__(self, image, annotations=None, masks=None, meta=None):
        """

        :param image: A ndarray matrix.
        :param annotations: A ndarray of points, with shape (number_of_annotations, number_of_points, 2)
        :param masks: A ndarray matrix with shape equal to image.
        :param meta: A dictionary with additional metadata.
        """

        if annotations is not None:
            assert annotations.ndim == 3 and annotations.shape[2] == 2, \
                ("Invalid annotations shape. The required format is a ndarray "
                 "of shape equal to (number_of_annotations, number_of_points, 2).")
            assert isinstance(annotations, np.ndarray), \
                "Annotations should be a np.ndarray instance."

        self.image = image
        self.annotations = annotations
        self.masks = masks
        self.meta = meta


class LenaSample(Sample):
    """Sample with Lena image for debugging purposes. """
    def __init__(self):
        super().__init__(read_image('lena.jpg'))
