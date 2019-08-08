import unittest
import aug


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
                self.assertEquals(input_sample.image.shape, output_sample.image.shape)
            else:
                self.assertNotEquals(input_sample.image.shape, output_sample.image.shape)


if __name__ == "__main__":
    unittest.main()
