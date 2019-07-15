import random
import cv2
import numpy as np
from aug import Operation, perform_randomly


@perform_randomly
class Contrast(Operation):

    def __init__(self, scale=.2):
        """ Adjust image contrast.

        :param scale: A coefficient determining intensity (Value < 1 will transform image to gray,
            while values > 1 will transform image to super-contrast).
        """
        self._scale = scale

    def apply_on_image(self, image):
        return np.clip(self._scale * (image - 128.0) + 128, 0, 255).astype(np.uint8)


@perform_randomly
class GaussNoise(Operation):

    def __init__(self, avg=0, std_dev=30):
        self._avg = avg
        self._std_dev = std_dev

    def apply_on_image(self, image):
        img_cpy = image.copy()
        try:
            m = tuple(image.shape[2] * [self._avg])
            s = tuple(image.shape[2] * [self._std_dev])
        except KeyError:
            m = self._avg
            s = self._std_dev

        cv2.randn(img_cpy, m, s)

        return cv2.add(img_cpy, image)


@perform_randomly
class PepperNoise(Operation):

    def __init__(self, percent=0.0005, value=None):
        self._value = value
        self._percent = percent

    def apply_on_image(self, image):
        channels_num = image.shape[2]
        if self._value is None:
            self._value = [0] * channels_num if channels_num < 4 else [0, 0, 0, 255]

        points_num = int(self._percent * image.size)
        for _ in range(points_num):
            image[random.randint(0, image.shape[0] - 1),
                  random.randint(0, image.shape[1] - 1), :] = self._value

        return image


@perform_randomly
class SaltNoise(Operation):

    def __init__(self, percent=0.0005, value=None):
        self._value = value
        self._percent = percent

    def apply_on_image(self, image):
        channels_num = image.shape[2]
        if self._value is None:
            self._value = [255] * channels_num

        return PepperNoise(self._percent, self._value, p=1.).apply_on_image(image)


@perform_randomly
class JpegNoise(Operation):

    def __init__(self, quality=0.1):
        self._quality = int(100 * quality)

    def apply_on_image(self, image):
        _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])

        return cv2.imdecode(buff, cv2.IMREAD_COLOR)


@perform_randomly
class Pixelize(Operation):

    def __init__(self, ratio=.2):
        self._ratio = ratio

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]
        tmp_w, tmp_h = int(im_width * self._ratio), int(im_height * self._ratio)

        image = cv2.resize(image, (tmp_w, tmp_h), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_NEAREST)

        return image


@perform_randomly
class Gamma(Operation):

    def __init__(self, gamma=.5, a=1):
        self._gamma = gamma
        self._a = a

    def apply_on_image(self, img):
        if img.dtype == np.uint8:
            inverted_gamma = 1.0 / self._gamma
            table = np.array([((i / 255.0)**inverted_gamma) * 255 for i in np.arange(0, 256)])

            table = table.astype(np.uint8)
            img = cv2.LUT(img, table)

        else:
            img = np.power(img, self._gamma)

        return self._a * img


@perform_randomly
class ChannelShuffle(Operation):

    def apply_on_image(self, image):
        assert image.shape[2] in [3, 4]
        ch_arr = [0, 1, 2]
        random.shuffle(ch_arr)
        image = image[..., ch_arr]
        return image


@perform_randomly
class Inversion(Operation):

    def apply_on_image(self, image):
        image[:, :, :3] = 255 - image[:, :, :3]
        return image


@perform_randomly
class Clahe(Operation):

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size

    def apply_on_image(self, img):
        assert (img.dtype == np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self._clip_limit, tileGridSize=self._tile_grid_size)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return img
