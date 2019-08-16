import random

import cv2
import numpy as np
from aug import Operation, perform_randomly
from aug import SaltNoise
from aug.ops import utils


@perform_randomly
class Erosion(Operation):

    def __init__(self, kernel_size=5, reversed=False):
        self._kernel_size = kernel_size
        self._reversed = reversed

    def apply_on_image(self, image):
        self._kernel_size = self._kernel_size if self._kernel_size % 2 != 0 else self._kernel_size + 1
        kernel = np.ones((self._kernel_size, self._kernel_size), np.uint8)
        er, dil = cv2.erode, cv2.dilate
        if self._reversed:
            er, dil = dil, er

        image[:, :, 0] = er(image[:, :, 0], kernel, iterations=1)
        image[:, :, 1] = er(image[:, :, 1], kernel, iterations=1)
        image[:, :, 2] = er(image[:, :, 2], kernel, iterations=1)
        if image.shape[2] > 3:
            image[:, :, 3] = dil(image[:, :, 3], kernel, iterations=1)

        return image


@perform_randomly
class Dilatation(Operation):

    def __init__(self, kernel_size=3):
        self._kernel_size = kernel_size

    def apply_on_image(self, image):
        return Erosion(kernel_size=self._kernel_size, reversed=True).apply_on_image(image)


class BoundingBoxesFinder(object):
    """ Find bounding boxes of letters. """

    def apply_on_image(self, in_image):
        last_empty = None
        last_letter = None
        top_border = None
        bottom_border = None
        borders = []

        image = in_image.copy()
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        im_height, im_width = image.shape[:2]

        # Find top/bottom border
        for i in range(im_height):
            column_sum = sum(image[i, :])
            if column_sum != 255 * im_width and top_border is None:
                top_border = i

            column_sum = sum(image[-i, :])
            if column_sum != 255 * im_width and bottom_border is None:
                bottom_border = im_height - i

            if top_border is not None and bottom_border is not None:
                break

        # Find vertical borders
        for i in range(im_width):
            column_sum = sum(image[:, i])
            if column_sum != 255 * im_height:
                if last_letter != i - 1:
                    borders.append(i)
                last_letter = i
            else:
                if last_empty is not None and last_empty != i - 1:
                    borders.append(i)
                last_empty = i

        vertical_borders = sorted(borders)
        crop_borders = []

        for i in range(len(vertical_borders), 2):
            crop_borders.append(
                [top_border, bottom_border, vertical_borders[i], vertical_borders[i + 1]])

        return crop_borders


class SeparatedLettersErosionOrDilatation:
    EROSION_MODE = 0
    DILATATION_MODE = 1
    MIX_MODE = 2

    # padding - distance between countour and box borders
    def __init__(self,
                 mode=MIX_MODE,
                 padding=6,
                 iterations=(1, 6),
                 kernel_size=(5, 5),
                 salt_noise=True):
        assert mode in [self.EROSION_MODE, self.DILATATION_MODE, self.MIX_MODE]

        self._padding = padding
        self._mode = mode
        self._iterations = iterations
        self._kernel_size = kernel_size
        self._salt_noise = salt_noise

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]

        if self._salt_noise:
            image = SaltNoise(p=1., percent=random.uniform(0.0001, 0.001)).apply_on_image(image)

        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 75, 150, 3)
        thresh = 255 - thresh
        img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        dst = image

        for i in range(len(contours)):
            mask = np.zeros_like(img)
            cv2.drawContours(mask, contours, i, 255, -1)

            x, y = np.where(mask == 255)
            topx, topy = np.min(x), np.min(y)
            bottomx, bottomy = np.max(x), np.max(y)

            out = image[topx - self._padding:bottomx + self._padding, topy - self._padding:bottomy +
                        self._padding]
            # out = 255 - out

            kernel = cv2.getStructuringElement(
                random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS, cv2.MORPH_RECT]),
                self._kernel_size)

            if not self._mode == self.MIX_MODE:
                if self._mode == self.EROSION_MODE:
                    transformed = cv2.erode(out,
                                            kernel,
                                            iterations=random.randint(*self._iterations))

                elif self._mode == self.DILATATION_MODE:
                    transformed = cv2.dilate(out,
                                             kernel,
                                             iterations=random.randint(*self._iterations))
                else:
                    raise Exception('Unknown mode')
            else:
                if random.randint(0, 1):
                    transformed = cv2.erode(out,
                                            kernel,
                                            iterations=random.randint(*self._iterations))
                else:
                    transformed = cv2.dilate(out,
                                             kernel,
                                             iterations=random.randint(*self._iterations))

            transformed = 255 - transformed

            dst[topx - self._padding:bottomx + self._padding, topy - self._padding:bottomy +
                self._padding] = transformed

        dst = cv2.resize(dst, (im_width, im_height), interpolation=cv2.INTER_CUBIC)

        return dst


@perform_randomly
class ScatterLetters(Operation):

    def __init__(self, max_dev_ox=0.02, max_dev_oy=0.15):
        self._max_devx = max_dev_ox
        self._max_devy = max_dev_oy

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]
        fill_color = (255, 255, 255, 0)

        h = int(self._max_devy * im_height + 1)
        w = int(self._max_devx * im_width + 1)
        image = cv2.copyMakeBorder(image, h, h, w, w, cv2.BORDER_CONSTANT, value=fill_color)
        borders = BoundingBoxesFinder().apply_on_image(image)

        for b in borders:
            y1, y2, x1, x2 = b
            ox_dev = int(random.uniform(-self._max_devx, self._max_devx) * im_width) / 2
            oy_dev = int(random.uniform(-self._max_devy, self._max_devy) * im_height) / 2

            tmp_x1, tmp_x2 = x1 + ox_dev, x2 + ox_dev
            tmp_y1, tmp_y2 = y1 + oy_dev, y2 + oy_dev

            tmp_tensor = image[y1:y2, x1:x2].copy()
            image[max(0, y1 - 1):min(image.shape[0], y2 + 1),
                  max(0, x1 - 1):min(image.shape[1], x2 + 1)] = fill_color
            image[tmp_y1:tmp_y2, tmp_x1:tmp_x2] = tmp_tensor

        return cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_CUBIC)


@perform_randomly
class Noise(Operation):

    def __init__(self, mode='normal'):
        self._mode = mode
        assert self._mode in ['dotted', 'normal']

    def noise(self, mask, image, color_diff=10, percent=0.05, radius=10):
        im_height, im_width = image.shape[:2]
        tmp = image.copy()
        tmp = 255 - tmp
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGBA2GRAY)
        _, tmp = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)

        number = int(percent * im_height * im_width)
        for _ in range(number):
            c = random.randint(0, color_diff)
            color = [c, c, c, 255]

            oy = random.randint(0, im_height - 1)
            ox = random.randint(0, im_width - 1)
            if mask[oy, ox]:
                cv2.circle(image, (ox, oy), 0, color, radius)

        return image

    def apply_noises(self, img, configs):
        mask = img.copy()
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
        mask = 255 - mask

        img2 = np.zeros(img.shape, dtype=np.uint8)
        img2[:, :, :3] = 255

        config = random.choice(configs)
        for params in config:
            img2 = self.noise(mask, img2, *params)

        return img2

    def apply_noises_dotted_font(self, img):
        """ Apply different kinds of noises defined in configs (dotted fonts)
            Single config row:
                [x, y, z]
            x - max deviation from base color
            y - density of noise in percent
            z - radius of single dot
        """
        configs = [[
            [20, 2.2, 1],
        ]]

        return self.apply_noises(img, configs)

    def apply_noises_normal_font(self, img):
        """ Apply different kinds of noises defined in configs (normal fonts)
            Single config row:
                [x, y, z]
            x - max deviation from base color
            y - density of noise in percent
            z - radius of single dot
        """
        configs = [[
            [20, 0.7, 1],
            [100, 0.01, 7],
            [70, 0.05, 4],
        ], [
            [20, 0.25, 3],
            [40, 0.2, 2],
            [130, 0.01, 2],
        ], [
            [20, 2.2, 1],
        ]]

        return self.apply_noises(img, configs)

    def apply_on_image(self, image):
        if self._mode == 'normal':
            return self.apply_noises_normal_font(image)

        if self._mode == 'dotted':
            return self.apply_noises_dotted_font(image)


@perform_randomly
class RandomSizeBorder(Operation):

    def __init__(self,
                 max_border=.1,
                 horizontal_sides_probability=.5,
                 vertical_sides_probability=.5):
        self._max_border = max_border
        self._horizontal_sides_probability = horizontal_sides_probability
        self._vertical_sides_probability = vertical_sides_probability

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]

        borders = [
            0 if random.random() < self._horizontal_sides_probability else int(
                random.uniform(0., self._max_border * im_height)) for _ in range(2)
        ]
        borders.extend([
            0 if random.random() < self._vertical_sides_probability else int(
                random.uniform(0., self._max_border * im_width)) for _ in range(2)
        ])

        image = cv2.copyMakeBorder(image,
                                   *borders,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255, 0))

        return cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_CUBIC)


@perform_randomly
class HorizontalCut(Operation):

    def __init__(self, left=.1, right=.1, rescale=True, horizontal=True):
        self._left = left
        self._right = right
        self._rescale = rescale
        self._horizontal = horizontal

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]

        if self._horizontal:
            left = int(im_width * self._left)
            right = int(im_width * self._right)
            image = image[:, left:im_width - right]
        else:
            top = int(im_height * self._left)
            bottom = int(im_height * self._right)
            image = image[top:im_height - bottom, :]

        if self._rescale:
            image = cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_CUBIC)

        return image


@perform_randomly
class VerticalCut(Operation):

    def __init__(self, top=.1, bottom=.1, rescale=True):
        self._top = top
        self._bottom = bottom
        self._rescale = rescale

    def apply_on_image(self, image):
        return HorizontalCut(self._top, self._bottom, self._rescale,
                             horizontal=False).apply_on_image(image)


@perform_randomly
class Scratches(Operation):
    """
       Scratches will be drawn only in box witch coords: left_top_corner=(min_x, min_y)
       and right_bottom_corner=(max_x, max_y) if corods will be not set, the scratches
       will be drawn on whole image
    """

    def __init__(self, num_scratches=20, alpha=None):
        self._min_x = None
        self._max_x = None
        self._min_y = None
        self._max_y = None
        self._num_scratches = num_scratches
        self._alpha = alpha if alpha is not None else .5

    def test_probability(self, prob):
        n = random.randint(0, 100)
        return n <= prob

    def apply_on_image(self, image):
        h, w = image.shape[:2]

        min_x, min_y = 0, 0
        max_x, max_y = 2 * w, 2 * h

        scratches = np.zeros((max_y, max_x, 3), np.uint8)
        scratches[:] = 0

        # main scratch
        for i in range(0, self._num_scratches):
            x1 = random.randint(min_x, max_x)
            x2 = random.randint(min_x, max_x)
            y1 = random.randint(min_y, max_y)
            y2 = random.randint(min_y, max_y)

            color = tuple([random.randint(0, 255)] * 3)

            cv2.line(scratches, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)

            # additional scratches for main scratch
            num_additional_scratches = random.randint(1, 4)
            for j in range(0, num_additional_scratches):
                if self.test_probability(35):
                    new_color = random.randint(15, 70)

                    param_x1 = random.randint(1, 5)
                    param_x2 = random.randint(1, 5)
                    param_y1 = random.randint(1, 5)
                    param_y2 = random.randint(1, 5)
                    cv2.line(scratches, (x1 - param_x1, y1 - param_x2),
                             (x2 - param_y1, y2 - param_y2), (new_color, new_color, new_color),
                             thickness=1,
                             lineType=cv2.LINE_AA)

        top, bottom = h // 2, scratches.shape[0] - (h - h // 2)
        left, right = w // 2, scratches.shape[1] - (w - w // 2)

        scratches = scratches[top:bottom, left:right]
        dst = cv2.addWeighted(image[:, :, :3], 1.0, scratches, self._alpha, 0.0)

        return cv2.resize(dst, (w, h), interpolation=cv2.INTER_CUBIC)


@perform_randomly
class TextureModification(Operation):
    """
    Creates effect of dirt/dust.
    """

    def __init__(self, blur_kernel=(3, 3), emboss_kernel_size=None, alpha=None):
        self._blur_kernel = blur_kernel
        self._emboss_kernel_size = random.choice([9, 11]) if \
            emboss_kernel_size is None else emboss_kernel_size
        self._alpha = random.uniform(0.4, 0.7) if alpha is None else alpha

    def apply_on_image(self, image):

        def create_emboss_kernel_top_down(size):
            assert size % 2 == 1, "Kernel must be of an uneven size!"
            k = np.ones((size, size), dtype=np.int)
            for i in range(size):
                for j in range(size):
                    k[i][j] = -1
                    if i > (size - 1) / 2:
                        k[i][j] = 1
                    if i == (size - 1) / 2:
                        k[i][j] = 0
            return k

        h, w = image.shape[:2]
        k_size = max(int((h + w) // 300), 3)
        self._blur_kernel = k_size, k_size

        # creating 'dirt'
        random_noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        dirt_kernel = create_emboss_kernel_top_down(self._emboss_kernel_size)
        dirt_colour = cv2.filter2D(random_noise, -1, dirt_kernel)
        gray_dirt = cv2.cvtColor(dirt_colour, cv2.COLOR_BGR2GRAY)

        # back to 3 channels (can't use addWeighted() to add images that have different number of channels)
        gray_dirt_3_channels = cv2.cvtColor(gray_dirt, cv2.COLOR_GRAY2BGR)

        blurred_dirt = cv2.blur(gray_dirt_3_channels, self._blur_kernel)
        blurred_dirt = utils.unify_num_of_channels(image, blurred_dirt)
        final = cv2.addWeighted(image, 1.0, blurred_dirt, self._alpha, 0.0)
        return final


@perform_randomly
class Jitter(Operation):

    def __init__(self, magnitude=.25):
        super().__init__()
        self._magnitude = magnitude

    def apply_on_image(self, image):
        if image.ndim == 3:
            w, h, c = image.shape[:3]
        else:
            w, h = image.shape[:2]
            c = 1

        magnitude = int(min(w, h) / 10 * self._magnitude)
        noise_x = np.random.randint(magnitude, size=w * h) - magnitude // 2
        noise_y = np.random.randint(magnitude, size=w * h) - magnitude // 2

        indices_x = np.clip(noise_x + np.arange(w * h), 0, w * h - 1)
        indices_y = np.clip(noise_y + np.arange(w * h), 0, w * h - 1)

        image = image[:, :].reshape(w * h, c)[indices_x].reshape(h, w, c)
        image = np.transpose(image, (1, 0, 2))
        image = image[:, :].reshape(w * h, c)[indices_y].reshape(h, w, c)
        image = np.transpose(image, (1, 0, 2))

        return image
