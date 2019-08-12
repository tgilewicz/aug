from __future__ import print_function
import random
import cv2
from enum import Enum
import numpy as np

from aug.ops import utils
from aug import Operation, perform_randomly

ITERATIONS_LIMITER = 1000


class Direction(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


def get_cut_letter_coords(dir_id, border, shape, b_range):
    border = int(border * random.uniform(*b_range))

    if dir_id.name == Direction.TOP.name:
        return [(0, 0), (0, border), (shape[1] - 1, border), (shape[1] - 1, 0)]
    elif dir_id.name == Direction.BOTTOM.name:
        return [(0, shape[0] - 1 - border), (shape[1] - 1, shape[0] - 1 - border),
                (shape[1] - 1, shape[0] - 1), (0, shape[0] - 1)]
    elif dir_id.name == Direction.LEFT.name:
        return [(0, 0), (border, 0), (border, shape[0] - 1), (0, shape[0] - 1)]
    elif dir_id.name == Direction.RIGHT.name:
        return [(shape[1] - 1 - border, 0), (shape[1] - 1, 0), (shape[1] - 1, shape[0] - 1),
                (shape[1] - 1 - border, shape[0] - 1)]


def fit_image_to_available_area(cut_text, dir_id, coordinates):
    height = abs(coordinates[0][1] - coordinates[2][1])
    width = abs(coordinates[0][0] - coordinates[2][0])

    if dir_id.name == Direction.TOP.name:
        return cut_text[cut_text.shape[0] - 1 - height:cut_text.shape[0] - 1, 0:width]

    elif dir_id.name == Direction.BOTTOM.name:
        return cut_text[0:height, 0:width]

    elif dir_id.name == Direction.LEFT.name:
        return cut_text[0:height, cut_text.shape[1] - 1 - width:cut_text.shape[1] - 1]

    elif dir_id.name == Direction.RIGHT.name:
        return cut_text[0:height, 0:width]


def get_random_offsets(direction, coords, crop):
    x_offset = 0
    y_offset = 0

    if direction.name == Direction.TOP.name or direction.name == Direction.BOTTOM.name:
        x_diff = int(abs(coords[0][0] - coords[3][0])) - crop.shape[1]
        if x_diff > 0:
            x_offset = random.randint(0, x_diff)

    if direction.name == Direction.LEFT.name or direction.name == Direction.RIGHT.name:
        y_diff = int(abs(coords[0][1] - coords[3][1])) - crop.shape[0]
        if y_diff > 0:
            y_offset = random.randint(0, y_diff)

    return x_offset, y_offset


class TemplateContour(object):
    """ Load contour from image file and insert in a random position """

    def __init__(self, contour, direction=None, contour_width=(.6, .9), margin=None):
        self._contour = contour
        self._direction = direction
        self._contour_width = contour_width
        self._margin = margin

    def apply_on_image(self, image):
        h = int(random.uniform(.8, 2) * image.shape[0])
        w = int(self._contour.shape[1] / (self._contour.shape[0] / float(h)))
        symbol = cv2.resize(self._contour, (w, h), interpolation=cv2.INTER_CUBIC)

        if self._margin is None:
            self._margin = int(2 * image.shape[0])

        if self._direction is None:
            self._direction = Direction(random.choice([2, 3]))

        image = cv2.copyMakeBorder(image,
                                   0,
                                   0,
                                   self._margin,
                                   self._margin,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255, 0))

        coords = get_cut_letter_coords(self._direction, self._margin, image.shape,
                                       self._contour_width)
        symbol = fit_image_to_available_area(symbol, self._direction, coords)

        x_offset, y_offset = get_random_offsets(self._direction, coords, symbol)

        image[y_offset + coords[0][1]:y_offset + coords[0][1] + symbol.shape[0], x_offset +
              coords[0][0]:x_offset + coords[0][0] + symbol.shape[1]] = symbol

        return utils.fit_borders(image)


@perform_randomly
class RandomCurveContour(Operation):

    def __init__(self, color=None, limit=500, iterations=10):
        self._color = color
        self._limit = limit
        self._iterations = iterations

    @staticmethod
    def draw_points(image, last, color, limit):
        last_x, last_y = last
        for _ in range(limit):
            image[last_y, last_x, :] = color

            for _ in range(ITERATIONS_LIMITER):
                x, y = random.choice([(last_x - 1, last_y), (last_x, last_y - 1),
                                      (last_x + 1, last_y), (last_x, last_y + 1)])

                if 0 < x < image.shape[1] and 0 < y < image.shape[0]:
                    last_x, last_y = x, y
                    break

    def apply_on_image(self, image):
        image = image.copy()

        if self._color is None:
            self._color = [255, 255, 255]

        for _ in range(self._iterations):
            x, y = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
            self.draw_points(image, (x, y), self._color, self._limit)

        return image


@perform_randomly
class RandomRadialDirt(Operation):
    """ Draw random radial contour """

    def __init__(self, max_radius=5):
        self._max_radius = max_radius

    def apply_on_image(self, image):
        image = image.copy()

        repeated_dirt_probability = 0.5
        while True:
            center = (random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1))

            color = utils.random_dark_color()
            cv2.circle(image, center, random.randint(0, self._max_radius), color, cv2.FILLED)

            if random.random() < repeated_dirt_probability:
                return image


@perform_randomly
class RandomShapeShadow(Operation):

    def __init__(self, shapes=('ellipse', 'polygon'), iterations=4, max_color=100):
        self._shapes = shapes
        self._iterations = iterations
        self._max_color = 255 - max_color

    def random_ellipse(self, image):
        h, w = image.shape[:2]
        center_x = random.randint(0, w - 1)
        center_y = random.randint(0, h - 1)

        axis_x = random.randint(0, min(w, h) // 2 - 1)
        axis_y = random.randint(0, min(w, h) // 2 - 1)

        angle = random.randint(0, 360)
        angles = [random.randint(0, 360), random.randint(0, 360)]
        angle_start = min(angles)
        angle_end = max(angles)
        color = random.randint(0, self._max_color)

        return cv2.ellipse(image, (center_x, center_y), (axis_x, axis_y), angle, angle_start,
                           angle_end, color, cv2.FILLED)

    def random_polygon(self, image):
        h, w = image.shape[:2]
        vrx_number = random.randint(3, 5)

        vrx = []
        for _ in range(vrx_number):
            vrx.append([random.randint(0, w - 1), random.randint(0, h - 1)])
        vrx = np.array(vrx, dtype=np.int32)

        return cv2.fillPoly(image, pts=[vrx], color=random.randint(0, self._max_color))

    def parse_args(self):
        ops = []

        if 'ellipse' in self._shapes:
            ops.append(self.random_ellipse)

        if 'polygon' in self._shapes:
            ops.append(self.random_polygon)

        return ops

    def apply_on_image(self, image):
        ops = self.parse_args()
        h, w = image.shape[:2]
        min_dim = min(h, w)
        mask = np.zeros(shape=(2 * h, 2 * w), dtype=np.uint16)

        for _ in range(self._iterations):
            op = random.choice(ops)
            mask_tmp = op(mask.copy())
            k = random.randint(1, int(.25 * min_dim))
            mask_tmp = cv2.blur(mask_tmp, ksize=(k, k))
            mask += mask_tmp

        mask = np.clip(mask, 0, self._max_color)

        top, bottom = h // 2, mask.shape[0] - (h - h // 2)
        left, right = w // 2, mask.shape[1] - (w - w // 2)

        mask = mask[top:bottom, left:right]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask = utils.unify_num_of_channels(image, mask)

        diff = image.astype(np.int32) - mask.astype(np.int32)
        img = np.clip(diff, 0, 255).astype(np.uint8)

        return img


class CropRandomSideToContour(object):

    def __init__(self):
        self._side = Direction(random.randint(0, 3))

    def get_border(self, image):
        tmp = image[:, :, 3].copy()
        # tmp = 255 - tmp
        # _,  tmp = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)

        oy_sum = np.sum(tmp, axis=1)
        oy_not_zeros = np.where(oy_sum != 0)[0]

        ox_sum = np.sum(tmp, axis=0)
        ox_not_zeros = np.where(ox_sum != 0)[0]

        b = 0
        if self._side == Direction.TOP:
            return max(oy_not_zeros[0] - b, 0)
        elif self._side == Direction.BOTTOM:
            return min(oy_not_zeros[-1] + b, image.shape[0] - 1)
        elif self._side == Direction.LEFT:
            return max(ox_not_zeros[0] - b, 0)
        elif self._side == Direction.RIGHT:
            return min(ox_not_zeros[-1] + b, image.shape[1] - 1)

    def apply_on_image(self, image, border=None):
        """
            Convert Rgb image to gray scale, crop random side to contour.
            Contour is pixel of group of pixels object with brightness value less
            than 127 in gray scale.
        Args:
            image: RGB image
            border: size of margin to cut

        Returns:
            Cut image

        """
        border = border if border is not None else self.get_border(image)

        if self._side == Direction.TOP:
            image = image[border:, :]
        elif self._side == Direction.BOTTOM:
            image = image[:border, :]
        elif self._side == Direction.LEFT:
            image = image[:, border:]
        elif self._side == Direction.RIGHT:
            image = image[:, :border]

        return image

    def apply_on_list(self, images):
        borders = [self.get_border(image) for image in images]
        if random.random() < .0:
            # Get best margin to fit all characters properly
            b = min(borders) if self._side in [Direction.TOP, Direction.LEFT] else max(borders)
        else:
            # Get random margin (it is possible to cut too big part of character)
            b = random.choice(borders)

        images = [self.apply_on_image(image, border=b) for image in images]
        return images


@perform_randomly
class RandomEdge(Operation):

    @staticmethod
    def get_edge_points(shape):
        points = []
        if random.getrandbits(1):
            # horizontal
            points.extend([(0, shape[0] - 1), (shape[1] - 1, shape[0] - 1)])
            height = random.randint(0, shape[0] - 1)
            points.append((shape[1] - 1, height))
            points.append((0, height))
        else:
            # vertical
            points.extend([(0, 0), (0, shape[0] - 1)])
            width = random.randint(0, shape[1] - 1)
            points.append((width, shape[0] - 1))
            points.append((width, 0))

        return np.array([points])

    def apply_on_image(self, image):
        image = image.copy()

        points = self.get_edge_points(image.shape)
        bg_color = utils.random_bright_color()
        cv2.fillPoly(image, points, bg_color)

        return image


@perform_randomly
class CutOut(Operation):

    def __init__(self, size_range=(.1, .2), iterations=2):
        self._size_norm = random.uniform(*size_range)
        self._x_norm = 1. - self._size_norm
        self._y_norm = 1. - self._size_norm

        self._iterations = iterations

    def apply_on_image(self, image):
        h, w = image.shape[:2]
        image = image.copy()

        for i in range(self._iterations):
            size = int(min(h, w) * self._size_norm)

            x, y = int(w * self._x_norm), int(h * self._y_norm)
            image[y:y + size, x:x + size] = 0

        return image
