import unittest
import aug
from copy import deepcopy
import numpy as np


def get_available_operations():
    """Scan the module and return all available operations."""
    ops = []
    for name in dir(aug):
        element = getattr(aug, name)
        if isinstance(element, type) and issubclass(element, (aug.BaseWrapper, aug.Operation)) and element is not aug.BaseWrapper:
            ops.append(element)

    return ops


class TestOps(unittest.TestCase):
    def test_input_and_output_size_equality(self):
        resizable_ops = (aug.Pad, aug.PadToMultiple, aug.RotationWithBound, aug.Stretch)

        for op_class in get_available_operations():
            input_sample = aug.LenaSample()

            op = op_class()
            output_sample = op.apply(input_sample)

            # if issubclass(op_class, (aug.BaseWrapper, )):
            #     print(op._wrapped)

            if not issubclass(op_class, resizable_ops):
                self.assertEqual(input_sample.image.shape, output_sample.image.shape)
            else:
                self.assertNotEqual(input_sample.image.shape, output_sample.image.shape)

    def test_if_input_images_are_not_modified(self):
        for op_class in get_available_operations():
            input_sample = aug.LenaSample()
            input_sample_tmp = deepcopy(input_sample)

            op = op_class()
            _ = op.apply(input_sample)

            # if issubclass(op_class, (aug.BaseWrapper, )):
            #     print(op._wrapped)

            self.assertTrue(np.array_equal(input_sample_tmp.image, input_sample.image))
            self.assertTrue(np.array_equal(input_sample_tmp.annotations, input_sample.annotations))
            self.assertTrue(np.array_equal(input_sample_tmp.masks, input_sample.masks))

    def test_if_operations_are_deterministic(self):
        for op_class in get_available_operations():
            input_sample = aug.LenaSample()

            op = op_class()
            sample1 = op.apply(input_sample)
            sample2 = op.apply(input_sample)

            if issubclass(op_class, (aug.BaseWrapper, )):
                print(op._wrapped)

            self.assertTrue(np.array_equal(sample1.image, sample2.image))
            self.assertTrue(np.array_equal(sample1.annotations, sample2.annotations))
            self.assertTrue(np.array_equal(sample1.masks, sample2.masks))


if __name__ == "__main__":
    unittest.main()
