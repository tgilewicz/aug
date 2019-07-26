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
