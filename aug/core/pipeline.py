import time
import cv2
import numpy as np
import random
from aug.core.sample import LenaSample
import aug

from copy import deepcopy

from multiprocessing.pool import ThreadPool


class Pipeline(object):
    """Base class for all pipelines of augmentation.

    Other pipelines should subclass this class.

    """

    def apply(self, sample):
        """Runs all ops in the pipeline. """
        raise NotImplementedError

    def apply_batched(self, samples, workers=None):
        """Runs all ops in the pipeline on batch of samples in parallel. """

        pool = ThreadPool() if workers is None else ThreadPool(workers)
        out_samples = pool.map(self.apply, samples)

        pool.close()
        pool.join()

        return out_samples

    def time(self, image):
        """Measure time needed to apply operations in pipeline. """
        start = time.clock()
        self.apply(image)
        return {self.__class__.__name__: time.clock() - start}

    def time_norm(self, image):
        """Measure time needed to apply operations in pipeline. Return normalized values. """
        times = self.time(image)
        factor = 1.0 / sum(times.values())
        for key in times:
            times[key] = times[key] * factor
        return times

    def __str__(self):
        return type(self).__name__

    def show(self, sample, annotations=True, masks=True):
        """Apply operations on sample and display results with masks and annotations. """
        assert isinstance(sample.image, np.ndarray)
        assert sample.image.shape[0] > 0 and sample.image.shape[1] > 0

        sample_orig = deepcopy(sample)
        sample = self.apply(sample)

        drawing_orig = Pipeline.draw_sample(sample_orig, annotations, masks)

        drawing = Pipeline.draw_sample(sample, annotations, masks)

        cv2.putText(drawing_orig, 'orig', (0, int(drawing_orig.shape[0] * .98)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(drawing, 'aug', (0, int(drawing_orig.shape[0] * .98)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("orig", drawing_orig)
        cv2.imshow("aug", drawing)
        cv2.waitKey()
        cv2.destroyAllWindows()


    @staticmethod
    def draw_sample(sample, annotations=True, masks=True):
        if sample.image.ndim == 2:
            sample.image = np.expand_dims(sample.image, axis=2)

        if sample.masks is not None and masks is True:
            for mask in list(sample.masks):
                mask = np.squeeze(mask)

                channel = random.randint(0, sample.image.shape[2] - 1)
                sample.image = sample.image.astype(np.int16)
                sample.image[:, :, channel] += \
                    random.randint(120, 150) * mask

                sample.image[:, :, channel] = np.clip(sample.image[:, :, channel], 0, 255)
                sample.image = sample.image.astype(np.uint8)

        if sample.annotations is not None and annotations is True:
            for anno in sample.annotations:
                for point in anno:
                    cv2.circle(sample.image, tuple(point),
                               max(2, int(.005 * min(sample.image.shape[:2]))), (0, 255, 0), -1)
        return sample.image


class TestPipeline(Pipeline):
    def __init__(self):
        self.seq = aug.Sequential(
            aug.Rotation90()
        )

    def apply(self, sample=LenaSample()):
        return self.seq.apply(sample)

    def show(self, sample=LenaSample()):
        sample = self.apply(sample)

        cv2.imshow("Image", sample.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
