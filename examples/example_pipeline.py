import cv2
import numpy as np
import aug


class SimpleExample(aug.Pipeline):

    def __init__(self):
        super(SimpleExample, self).__init__()
        self.seq = aug.Sequential(
            aug.Rotation(p=.5, angle=90),
            aug.GaussianBlur(p=1.),
        )

    def apply(self, sample):
        return self.seq.apply(sample)


class ComplexExamplePipeline(aug.Pipeline):

    def __init__(self):
        super(ComplexExamplePipeline, self).__init__()
        self.seq1 = aug.Sequential(
            self.affine_ops(),
            aug.Choice(
                aug.Stretch(p=.5, x_scale=aug.uniform(.25, .5), y_scale=aug.uniform(.25, .5)),
                aug.Rotation(p=.5, angle=aug.truncnorm(0., 5., 5., 10.))),
            aug.GaussianBlur(p=1),
        )

        self.seq2 = aug.Sequential(aug.GaussianBlur(p=1), aug.GaussianBlur(p=1))

    def affine_ops(self):
        return aug.Sequential(
            aug.Stretch(p=.5, x_scale=aug.uniform(.25, .5), y_scale=aug.uniform(.25, .5)),
            aug.Rotation(p=.5, angle=aug.truncnorm(0., 5., 5., 10.)))

    def apply(self, sample):
        sample = self.seq1.apply(sample)
        sample = self.seq2.apply(sample)

        return sample


def main():
    pipeline = SimpleExample()
    img = cv2.imread('lena.jpg')
    img = cv2.resize(img, dsize=(200, 200))
    img_orig = img.copy()

    sample = aug.Sample(img)

    print(pipeline.time(sample))
    sample = pipeline.apply(sample)

    cv2.imshow('lena.jpg', np.concatenate((img_orig, sample.image), axis=1))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
