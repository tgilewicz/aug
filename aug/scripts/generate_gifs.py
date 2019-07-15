import aug
import cv2
import imageio
import random

lena = cv2.resize(imageio.imread('../../images/lena.jpg'), (220, 220), cv2.INTER_AREA)


def generate_gif(file_name, images):
    imageio.mimsave('{}.gif'.format(file_name), images, duration=.5, fps=2, subrectangles=True)


class PepperAndSalt(aug.Pipeline):

    def __init__(self, percent):
        super().__init__()
        self._percent = percent

    def apply(self, sample):
        sample = aug.PepperNoise(percent=self._percent).apply(sample)
        sample = aug.SaltNoise(percent=self._percent).apply(sample)
        return sample


class Flip(aug.Pipeline):

    def apply(self, sample):
        return aug.HorizontalFlip(
            p=.5).apply(sample) if random.getrandbits(1) else aug.VerticalFlip(p=.5).apply(sample)


def photometric():
    generate_gif(
        "op_contrast",
        [aug.Contrast(scale=i).apply(aug.Sample(lena)).image for i in (.2, .6, .8, 1.2, 1.4, 1.8)])

    generate_gif(
        "op_gauss_noise",
        [aug.GaussNoise(std_dev=i).apply(aug.Sample(lena)).image for i in range(10, 60, 10)])

    generate_gif(
        "op_salt_pepper",
        [PepperAndSalt(percent=i).apply(aug.Sample(lena.copy())).image
         for i in (0.0005, 0.001, 0.005, 0.01)])

    generate_gif(
        "op_jpeg_noise",
        [aug.JpegNoise(quality=i / 10.).apply(aug.Sample(lena)).image
         for i in reversed(range(1, 11, 2))])

    generate_gif(
        "op_pixelize",
        [aug.Pixelize(ratio=i / 10.).apply(aug.Sample(lena)).image for i in reversed(range(1, 7))])

    generate_gif(
        "op_inversion",
        [lena, aug.Inversion().apply(aug.Sample(lena.copy())).image])

    generate_gif(
        "op_clahe",
        [lena] + [aug.Clahe(tile_grid_size=(2 ** i, 2 ** i)).apply(
            aug.Sample(lena)).image for i in range(1, 6)])

    generate_gif(
        "op_gamma",
        [aug.Gamma(gamma=i / 10.).apply(aug.Sample(lena)).image for i in reversed(range(1, 7))])

    generate_gif(
        "op_channel_shuffle",
        [aug.ChannelShuffle().apply(aug.Sample(lena)).image for _ in range(4)])

    generate_gif(
        "op_zoom",
        [aug.Zoom(margin=i / 10).apply(aug.Sample(lena)).image for i in range(1, 5)])


def affine():
    generate_gif(
        "op_flip",
        [Flip().apply(aug.Sample(lena)).image for _ in range(1, 10)])

    generate_gif(
        "op_rotation",
        [aug.RotationWithBound(angle=i, mode=cv2.BORDER_REFLECT_101, change_size=False).apply(
            aug.Sample(lena.copy())).image for i in range(0, 360, 30)])

    generate_gif(
        "op_rotation90",
        [aug.Rotation90(iterations=i).apply(aug.Sample(lena.copy())).image for i in range(4)])


def blurs():
    generate_gif(
        "op_median_blur", [aug.MedianBlur(ksize_norm=i / 100.).apply(
            aug.Sample(lena.copy())).image for i in range(6)])

    generate_gif(
        "op_motion_blur",
        [aug.MotionBlur(ksize_norm=.08).apply(aug.Sample(lena.copy())).image for _ in range(6)])

    generate_gif("op_gaussian_blur",
                 [aug.GaussianBlur(ksize_norm=i / 100.).apply(
                     aug.Sample(lena.copy())).image for i in range(6)])

    generate_gif(
        "op_variable_blur", [aug.VariableBlur(ksize_norm=.4, modes=('radial', 'linear')).apply(aug.Sample(
            lena.copy())).image for i in range(6)])


def contours():
    generate_gif(
        "op_cutout",
        [aug.CutOut(size_range=(.1, .2), iterations=i).apply(aug.Sample(lena.copy())).image
            for i in range(5)])

    generate_gif(
        "op_shadow",
        [aug.RandomShapeShadow(max_color=150).apply(
            aug.Sample(lena.copy())).image for i in range(6)])

    generate_gif(
        "op_random_curve",
        [aug.RandomCurveContour(color=(0, 0, 0), limit=i).apply(aug.Sample(lena.copy())).image
            for i in range(100, 1000, 200)])


def distortions():
    generate_gif(
        "op_erosion",
        [aug.Erosion(kernel_size=i).apply(aug.Sample(lena.copy())).image for i in range(1, 10, 2)])

    generate_gif(
        "op_dilatation",
        [aug.Dilatation(kernel_size=i).apply(aug.Sample(lena.copy())).image for i in range(1, 10, 2)])

    generate_gif(
        "op_texture",
        [aug.TextureModification(alpha=.1, blur_kernel=(50, 50), emboss_kernel_size=i).apply(
            aug.Sample(lena.copy())).image for i in range(1, 10, 2)])

    generate_gif(
        "op_jitter",
        [lena] + [aug.Jitter(magnitude=i).apply(aug.Sample(lena)).image for i in (.1, .2, .3, .4, .5)])

    generate_gif(
        "op_scratches",
        [lena] + [aug.Scratches(num_scratches=i, alpha=.2).apply(aug.Sample(lena.copy())).image
                  for i in range(10, 100, 20)])


def lighting():
    generate_gif(
        "op_global_brightness",
        [aug.GlobalBrightness(change=i).apply(aug.Sample(lena)).image
            for i in (.01, .03, .2, .8, .97, .99)])

    generate_gif(
        "op_radial_gradient",
        [aug.RadialGradient().apply(aug.Sample(lena)).image for _ in range(5)])

    generate_gif(
        "op_linear_gradient",
        [aug.LinearGradient().apply(aug.Sample(lena)).image for _ in range(5)])

    generate_gif(
        "op_halo",
        [aug.HaloEffect(alpha=.6, radius=aug.uniform(.3, .5)).apply(aug.Sample(lena)).image
            for _ in range(5)])

    generate_gif(
        "op_flashlight",
        [aug.Flashlight(alpha=aug.uniform(.2, .6), bg_darkness=aug.uniform(50, 150),
                        radius=aug.uniform(.3, .6)).apply(aug.Sample(lena)).image for _ in range(5)])


def blending():
    generate_gif(
        "op_blend",
        [aug.BlendWithRandomImage().apply(aug.Sample(lena)).image for _ in range(5)])


def perspective():
    generate_gif(
        "op_perspective",
        [aug.PerspectiveTransformation().apply(aug.Sample(lena)).image for _ in range(5)])

    generate_gif(
        "op_elastic",
        [aug.ElasticTransformation(alpha=aug.uniform(50., 200.), sigma=aug.uniform(2., 10.)).apply(
            aug.Sample(lena)).image for _ in range(6)])

    generate_gif(
        "op_grid",
        [aug.GridDistortion().apply(aug.Sample(lena)).image for _ in range(5)])

    generate_gif(
        "op_optical",
        [aug.OpticalDistortion(interpolation=cv2.BORDER_REFLECT).apply(aug.Sample(lena.copy())).image
            for _ in range(5)])


def main():
    photometric()
    affine()
    blurs()
    lighting()

    contours()
    distortions()
    blending()
    perspective()


if __name__ == "__main__":
    main()
