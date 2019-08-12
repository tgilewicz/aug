import io
import cv2
import numpy as np
import requests
from PIL import Image

from aug import Operation, perform_randomly, utils
from .utils import RingBuffer


@perform_randomly
class BlendWithRandomImage(Operation):
    """ Blend input image with random one downloaded from web. """
    URL = 'https://picsum.photos/{}/{}/?random'

    def __init__(self, ratio=.8):
        """
        Args:
            ratio: the weight of original image in blending operation
        """
        super().__init__()
        self._ratio = ratio
        self._random_image = None

    def apply_on_image(self, input_image):
        if self._random_image is not None:
            return cv2.addWeighted(input_image, self._ratio, self._random_image, 1 - self._ratio, 0)

        h, w = input_image.shape[:2]
        try:
            r = requests.get(self.URL.format(w, h), allow_redirects=True)
            f = io.BytesIO(r.content)
            random_img = Image.open(f)
            random_img = np.array(random_img)

            self._random_image = utils.unify_num_of_channels(input_image, random_img)

            return cv2.addWeighted(input_image, self._ratio, self._random_image, 1 - self._ratio, 0)

        except requests.exceptions.ConnectionError as e:
            print('Unable to download image. Error: {}'.format(e))
        except Exception as e:
            print('Unknown error occurred "{}"'.format(e))

        return input_image


@perform_randomly
class Pairing(Operation):
    """ Sample pairing operation
        (details: https://arxiv.org/pdf/1801.02929.pdf)
    """
    buffer = RingBuffer(size=100)

    def __init__(self, alpha=.2):
        super().__init__()
        self._alpha = alpha

    def apply_on_image(self, image):
        image_orig = image.copy()

        if len(Pairing.buffer) != 0:
            image_random = Pairing.buffer.get_sample()
            random_img = utils.unify_num_of_channels(image, image_random)

            image = cv2.addWeighted(image, 1 - self._alpha, random_img, self._alpha, 0)

        Pairing.buffer.append(image_orig)
        return image
