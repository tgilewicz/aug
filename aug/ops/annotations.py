import numpy as np

import aug


class AnnotationOrderFix(aug.Operation):
    """Transform points to the [left-top, right-bottom] order. """
    def apply_on_annotations(self, annotations):
        transformed = []
        for annotation in annotations:
            min_x, min_y = np.min(annotation, axis=0)
            max_x, max_y = np.max(annotation, axis=0)
            transformed.append([[min_x, min_y], [max_x, max_y]])

        return np.array(transformed)


class AnnotationOutsideImageFix(aug.Operation):
    def apply(self, sample):
        h, w = sample.image.shape[:2]

        sample.annotations[:, :, 1] = np.minimum(h, sample.annotations[:, :, 1])
        sample.annotations[:, :, 0] = np.minimum(w, sample.annotations[:, :, 0])

        sample.annotations[:, :, 1] = np.maximum(0, sample.annotations[:, :, 1])
        sample.annotations[:, :, 0] = np.maximum(0, sample.annotations[:, :, 0])

        diff_x = sample.annotations[:, 1, 0] - sample.annotations[:, 0, 0]
        diff_y = sample.annotations[:, 1, 1] - sample.annotations[:, 0, 1]

        indices_x = np.where(diff_x == 0)
        indices_y = np.where(diff_y == 0)
        indices = np.concatenate((indices_x[0], indices_y[0]), axis=0)

        if indices.size != 0:
            sample.annotations = np.delete(sample.annotations, indices, axis=0)

        return sample
