import cv2
import random
import numpy as np
import scipy
import math

from aug import Operation, perform_randomly, utils
import aug


@perform_randomly
class GlobalBrightness(Operation):

    def __init__(self, change=.5):
        """ Adjust global brightness.

        :param change: A coefficient determining intensity (Value 0.5 doesn't change brightness).
        """
        self._value = 32 * scipy.special.logit(np.array(change)).tolist()

    def apply_on_image(self, image):
        image = image.astype(np.float32)
        image += self._value
        return np.clip(image, 0, 255).astype(np.uint8)


@perform_randomly
class LinearGradient(Operation):

    def __init__(self, orientation="horizontal", edge_brightness=(.1, .3)):
        assert isinstance(edge_brightness, tuple) and len(edge_brightness) == 2, \
            "Argument edge_brightness should be a tuple with size 2."
        assert 0. < edge_brightness[0] < 1. and 0. < edge_brightness[1] < 1., \
            "Values of an edge_brightness argument should be in the [0, 1] range."
        assert orientation in ['horizontal', 'vertical'], "Unknown orientation value."

        self._orientation = orientation
        self._color1 = int(edge_brightness[0] * 255)
        self._color2 = int(edge_brightness[1] * 255)
        self._reverse = bool(random.getrandbits(1))

    def apply_on_image(self, image):
        image = np.int16(image)
        dim = image.shape[1] if self._orientation == "horizontal" else image.shape[0]
        for i in range(dim):
            coeff = i / float(dim)
            if self._reverse:
                coeff = 1. - coeff
            diff = int((self._color2 - self._color1) * coeff)
            if self._orientation == "horizontal":
                image[:, i, 0:3] = np.where(image[:, i, 0:3] + self._color1 + diff < 255,
                                            image[:, i, 0:3] + self._color1 + diff, 255)
            else:
                image[i, :, 0:3] = np.where(image[i, :, 0:3] + self._color1 + diff < 255,
                                            image[i, :, 0:3] + self._color1 + diff, 255)

        return image.astype(np.uint8)


@perform_randomly
class RadialGradient(Operation):

    def __init__(self,
                 inner_color=150,
                 outer_color=30,
                 center=None,
                 max_distance=None,
                 rect=False,
                 random_distance=False):
        """
            images: an input image
            center: the brightest point
            inner_color: color of the brightest point
            outer_color: color of the darkest point (localized in one of 4 corners)
            max_distance: distance between center and corner
        """
        self._inner_color = inner_color
        self._outer_color = outer_color
        self._center = center
        self._max_distance = max_distance
        self._rect = rect
        self._random_distance = random_distance

    @staticmethod
    def apply_radial(img, center, max_distance, inner_color, outer_color, rect=False):
        tmp = np.full(img.shape, outer_color, dtype=np.uint8)
        tmp_height, tmp_width = tmp.shape[:2]
        kernel = None

        left = max(0, 0 - (center[0] - max_distance))
        top = max(0, 0 - (center[1] - max_distance))
        right = max(0, (center[0] + max_distance) - tmp_width)
        bottom = max(0, (center[1] + max_distance) - tmp_height)
        tmp = cv2.copyMakeBorder(tmp, top, bottom, left, right, cv2.BORDER_CONSTANT)

        if rect:
            if random.getrandbits(1):
                dist = random.randint(10, int(.2 * tmp_width))
                cv2.rectangle(tmp, (center[0] - dist, 0), (center[0] + dist, tmp_height),
                              inner_color,
                              thickness=cv2.FILLED)
                k_size = dist if dist % 2 == 1 else dist - 1
                kernel = (k_size, 1)
            else:
                dist = random.randint(10, int(.2 * tmp_height))
                cv2.rectangle(tmp, (0, center[1] - dist), (tmp_width, center[1] + dist),
                              inner_color,
                              thickness=cv2.FILLED)
                k_size = dist if dist % 2 == 1 else dist - 1
                kernel = (1, k_size)
        else:
            cv2.circle(tmp, (center[0] + left, center[1] + top),
                       int(max_distance / 1.5),
                       inner_color,
                       thickness=cv2.FILLED)

        kernel = kernel if kernel else (max_distance, max_distance)
        tmp = cv2.blur(tmp, kernel, borderType=cv2.BORDER_CONSTANT)
        tmp = tmp[top:tmp.shape[0] - bottom, left:tmp.shape[1] - right]

        return np.clip(img.astype(np.uint16) + tmp.astype(np.uint16), 0, 255).astype(np.uint8)

    def apply_on_image(self, img):
        im_height, im_width, im_depth = img.shape
        inner_color = im_depth * [self._inner_color]
        outer_color = im_depth * [self._outer_color]

        if self._center is None:
            self._center = random.randint(0, im_height), random.randint(0, im_width)

        if not self._rect:
            if self._max_distance is None:
                if self._random_distance:
                    size = max(im_width, im_height)
                    self._max_distance = size * random.uniform(.1, .3)
                else:
                    self._max_distance = 0
                    corners = [(0, 0), (im_height, 0), (0, im_width), (im_height, im_width)]
                    for corner in corners:
                        distance = math.sqrt((corner[0] - self._center[0])**2 +
                                             (corner[1] - self._center[1])**2)
                        self._max_distance = max(distance, self._max_distance)

        return self.apply_radial(img,
                                 self._center,
                                 int(self._max_distance),
                                 inner_color,
                                 outer_color,
                                 rect=self._rect)


@perform_randomly
class CameraFlare(Operation):

    def __init__(self, alpha=0.8, radius=.5):
        assert 0 < radius <= 1
        self._alpha = alpha
        self._radius = radius

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]
        pos_x, pos_y = random.randint(0, im_width), random.randint(0, im_height)
        avg_dim = (im_height + im_width) / 2
        radius = int(avg_dim * self._radius)

        # white circle
        circle = np.zeros((im_height, im_width, 3), np.uint8)

        cv2.circle(circle, (pos_x, pos_y), radius, (255, 255, 255), -1)
        circle = cv2.blur(
            circle,
            (int(random.uniform(.15, .25) * avg_dim), int(random.uniform(.15, .25) * avg_dim)))

        circle = utils.unify_num_of_channels(image, circle)

        dst = cv2.addWeighted(image, 1.0, circle, self._alpha, 0.0)

        return cv2.resize(dst, (im_width, im_height), interpolation=cv2.INTER_CUBIC)


@perform_randomly
class HaloEffect(Operation):

    def __init__(self, radius=.5, alpha=0.8):
        self._radius = radius
        self._alpha = alpha

    def apply_on_image(self, image):
        h, w = image.shape[:2]
        avg_dim = (h + w) // 2
        self._radius = int(self._radius * avg_dim)

        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        halo_kernel = np.zeros((h, w), np.uint8)

        halo_kernel_radius = int(self._radius * random.uniform(.1, .35))
        cv2.circle(halo_kernel, (x, y), halo_kernel_radius, (255, 255, 255), -1)

        num_of_rays = 6
        b = halo_kernel_radius
        for _ in range(num_of_rays):
            offset_x = random.randint(b, int(2.5 * b))
            offset_y = random.randint(b, int(2.5 * b))

            offset_y = -offset_y if random.random() < .5 else offset_y
            offset_x = -offset_x if random.random() < .5 else offset_x

            cv2.line(halo_kernel, (x, y), (x + offset_x, y + offset_y), 255, 3)

        k1 = int(avg_dim * 0.1)
        halo_kernel = cv2.blur(halo_kernel, (k1, k1))
        halo_kernel = halo_kernel.astype(np.uint16)

        if random.random() < .5:
            halo_kernel += self.ring(x,
                                     y,
                                     h,
                                     w,
                                     avg_dim,
                                     k_size_norm=random.uniform(.1, .25),
                                     ring_thickness_norm=random.uniform(0.008, 0.015))

        if random.random() < .5:
            halo_kernel += self.ring(x,
                                     y,
                                     h,
                                     w,
                                     avg_dim,
                                     k_size_norm=random.uniform(.3, .5),
                                     ring_thickness_norm=random.uniform(0.05, 0.2))

        halo_kernel = np.clip(halo_kernel, 0, 255).astype(np.uint8)
        halo_kernel = cv2.cvtColor(halo_kernel, cv2.COLOR_GRAY2RGB)

        halo_kernel = utils.unify_num_of_channels(image, halo_kernel)

        dst = cv2.addWeighted(image, 1.0, halo_kernel, self._alpha, 0.0)

        return cv2.resize(dst, (w, h), interpolation=cv2.INTER_CUBIC)

    def ring(self, x, y, h, w, max_dim, k_size_norm, ring_thickness_norm):
        halo_ring = np.zeros((h, w), np.uint8)
        ring_thickness = int(max_dim * ring_thickness_norm)
        cv2.circle(halo_ring, (x, y), self._radius, 255, ring_thickness)

        k = int(max_dim * k_size_norm)
        k = k if k % 2 else k + 1
        return cv2.GaussianBlur(halo_ring.copy(), (k, k), 0)


@perform_randomly
class Flashlight(Operation):
    """
    Blackens whole image except from blurred circle with center in (x, y).
    :param alpha describes how strong torchlight should be
    :param darken_factor describes how strong we want to darken the image (we are decreasing `value` in hsv)
    """

    def __init__(self, alpha=0.6, bg_darkness=100, radius=.4):
        self._alpha = alpha
        self._bg_darkness = bg_darkness
        self._radius = radius

    def apply_on_image(self, image):
        im_height, im_width = image.shape[:2]
        pos_x = random.randint(int(1 / 4 * im_width), int(3 / 4 * im_width))
        pos_y = random.randint(int(1 / 4 * im_height), int(3 / 4 * im_height))
        min_wh = min(im_width, im_height)
        max_wh = max(im_width, im_height)
        radius = int(random.randint(min_wh, max_wh) * self._radius)

        k = random.uniform(1.5, 5.)
        blur_kernel_size = (int(radius / k), int(radius / k))

        def decrease_brightness(img, value=30):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = value
            v[v < lim] = 0
            v[v >= lim] -= value

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            return img

        darkened_image = decrease_brightness(image, value=self._bg_darkness)

        # create white circle on black background
        torchlight = np.zeros((im_height, im_width, 3), np.uint8)
        cv2.circle(torchlight, (pos_x, pos_y), radius, (255, 255, 255), -1)

        blurred_torchlight = cv2.blur(torchlight, blur_kernel_size)
        final_image = cv2.addWeighted(darkened_image, 1.0, blurred_torchlight, self._alpha, 0.0)
        return final_image


@perform_randomly
class Smudges(Operation):

    def __init__(self, number_of_smudges=None):
        self._number_of_smudges = number_of_smudges

    def apply_on_image(self, image):

        def add_smudge(img):
            h, w, _ = img.shape
            kernel_coef = random.randint(60, 90)
            ks = int((h * 1.25 + w) / kernel_coef)
            line_thickness_coef = random.randint(60, 80)
            line_thickness = int((1.15 * w + 2 * h) // line_thickness_coef)
            blur_kernel_size = (ks, ks)
            smudges = np.zeros((h, w, 3), np.uint8)
            point_height = random.randint(int(h / 10), int(9 * h / 10))
            p1 = (0, point_height)
            p2 = (w, point_height + random.randint(0, h // 25) * random.choice([-1, 1]))
            p1_b = (0, p1[1])
            p2_b = (w, p2[1])

            color = tuple([random.randint(100, 255)] * 3)
            cv2.line(smudges, p1, p2, color, line_thickness)
            cv2.line(smudges, p1_b, p2_b, color, 1)
            blurred_smudges = cv2.blur(smudges, blur_kernel_size)
            opacity = random.uniform(0.18, 0.25)

            blurred_smudges = utils.unify_num_of_channels(img, blurred_smudges)
            final_image = cv2.addWeighted(img, 1.0, blurred_smudges, opacity, 0.0)
            return final_image

        if self._number_of_smudges is None:
            self._number_of_smudges = random.randint(1, 6)

        flag = random.getrandbits(1)
        image = image if flag else aug.Transposition(p=1.).apply_on_image(image)

        for _ in range(self._number_of_smudges):
            image = add_smudge(image)

        image = image if flag else aug.Transposition(p=1.).apply_on_image(image)
        return image
