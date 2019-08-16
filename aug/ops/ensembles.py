""" Reusable ensembles of augmentation operations. """

import aug


@aug.perform_randomly
class Blurs(aug.Pipeline):

    def __init__(self):
        super().__init__()
        self.seq = aug.Choice(
            aug.MedianBlur(p=1., ksize_norm=aug.uniform(0., .006)),
            aug.GaussianBlur(p=1., ksize_norm=aug.uniform(.0, .05), sigma=aug.uniform(1, 5)),
            aug.MotionBlur(p=1., ksize_norm=aug.uniform(.01, .04)),
            aug.Choice(
                aug.VariableBlur(p=1., ksize_norm=aug.uniform(.015, .005), modes=('linear',)),
                aug.VariableBlur(p=1., ksize_norm=aug.uniform(.01, .005), modes=('radial',))))

    def apply(self, sample):
        return self.seq.apply(sample)

    def time(self, sample):
        return self.seq.time(sample)


@aug.perform_randomly
class Noises(aug.Pipeline):

    def __init__(self):
        super().__init__()
        self.seq = aug.Sequential(
            aug.GaussNoise(p=.45, avg=0, std_dev=aug.uniform(0, 40)),
            aug.SaltNoise(p=.2, percent=aug.uniform(0.0001, 0.0008)),
            aug.PepperNoise(p=.2, percent=aug.uniform(0.0001, 0.0008)),
            aug.CutOut(p=.1, size_range=(.05, .15), iterations=aug.uniform(1, 4)),
            aug.JpegNoise(p=.4, quality=aug.uniform(.1, .5)),
            aug.Pixelize(p=.4, ratio=aug.uniform(.2, .5)))

    def apply(self, sample):
        return self.seq.apply(sample)

    def time(self, sample):
        return self.seq.time(sample)


@aug.perform_randomly
class ColorAdjustment(aug.Pipeline):

    def __init__(self):
        super().__init__()
        self.seq = aug.Sequential(
            aug.Choice(
                aug.Brightness(p=1., change=aug.uniform(.01, .97)),
                aug.Choice(
                    aug.LinearGradient(p=1.,
                                       edge_brightness=(aug.uniform(.0, .05), aug.uniform(.1, .6)),
                                       orientation='horizontal'),
                    aug.LinearGradient(p=1.,
                                       edge_brightness=(aug.uniform(.0, .05), aug.uniform(.1, .6)),
                                       orientation='vertical'))),
            aug.Contrast(p=1., scale=aug.uniform(.5, 1.5)),
        )

    def apply(self, sample):
        return self.seq.apply(sample)

    def time(self, sample):
        return self.seq.time(sample)


@aug.perform_randomly
class Geometric(aug.Pipeline):

    def __init__(self):
        super().__init__()
        self.seq = aug.Sequential(
            aug.PerspectiveDistortion(p=.5, max_warp=.12),
            aug.Choice(
                aug.GridDistortion(num_steps=(10, 10), distort_limit=(.6, 1.4)),
                # TODO expensive computationally
                # aug.ElasticTransformation(p=.25, alpha=aug.uniform(20., 120.),
                #                           sigma=aug.uniform(8, 20),
                #                           alpha_affine_range=aug.uniform(8., 10.)),
            ),
            aug.Rotation(p=.25, angle=aug.uniform(-5, 5), mode='replicate'),
            aug.Zoom(p=.5, margin=.1),
        )

    def apply(self, sample):
        return self.seq.apply(sample)

    def time(self, sample):
        return self.seq.time(sample)
