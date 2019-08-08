__version__ = '0.1.0'


from aug.core.pipeline import Pipeline, TestPipeline
from aug.core.sequential import Sequential
from aug.core.shuffle import Shuffle
from aug.core.choice import Choice
from aug.core.operation import Operation
from aug.core.sample import Sample, LenaSample
from aug.core.range import *
from aug.core.decorators import *

from aug.ops.photometric import *
from aug.ops.affine import *
from aug.ops.distortions import *
from aug.ops.perspective import *
from aug.ops.lighting import *
from aug.ops.contours import *
from aug.ops.blending import *
from aug.ops.blurs import *
from aug.ops.ensembles import *
from aug.ops.other import *
from aug.ops.annotations import *


__all__ = [
    # Core
    "Choice",
    "LenaSample",
    "Operation",
    "Pipeline",
    "Sample",
    "Sequential",
    "Shuffle",
    "TestPipeline",

    "uniform",
    "rand_bool",
    "truncnorm",
    "perform_randomly",

    # Operations
    "BlendWithRandomImage",
    "Brightness",
    "CameraFlare",
    "ChannelShuffle",
    "Clahe",
    "Contrast",
    "CutOut",
    "Dilatation",
    "ElasticDistortion",
    "Erosion",
    "Flashlight",
    "Gamma",
    "GaussNoise",
    "GaussianBlur",
    "GridDistortion",
    "HaloEffect",
    "HorizontalFlip",
    "Inversion",
    "Jitter",
    "JpegNoise",
    "LinearGradient",
    "MedianBlur",
    "MotionBlur",
    "OpticalDistortion",
    "Pad",
    "PadToMultiple",
    "Pairing",
    "Pass",
    "PepperNoise",
    "PerspectiveDistortion",
    "Pixelize",
    "RadialGradient",
    "RandomCurveContour",
    "RandomShapeShadow",
    "Rotation",
    "Rotation90",
    "SaltNoise",
    "Scratches",
    "TextureModification",
    "VariableBlur",
    "VerticalFlip",
    "Zoom",

    # Ensembles
    "Blurs",
    "ColorAdjustment",
    "Geometric",
    "Noises",
]
