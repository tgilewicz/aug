<div align="center">

<img src="images/logo.png">

# AUG<em>-alpha</em>

[Warning, Unmaintained!] Alternatives: imgaug, albumentations, kornia.

[![PyPI Status](https://badge.fury.io/py/aug.svg)](https://badge.fury.io/py/aug)
[![PyPI Status](https://pepy.tech/badge/aug)](https://pepy.tech/project/aug)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

**AUG is an open source augmentation library based on OpenCV. The repository contains a set of image transformations for data augmentation and artificial data synthesis.** 


##### Major features:

* supports operations on masks and points,
* unified interface - most magnitude coefficients are in the range [0, 1],
* optimized operations,
* unique operations compared to imgaug/albumentations,
* Python 3, OpenCV 4.1.



## Installation

##### Pip:
```
pip install aug
```

##### The latest version, directly from github:
```bash
pip install -U git+https://github.com/cta-ai/aug
```

## Example operations
|   |   |   |   |
|---|---|---|---|
| Jitter | Radial gradient | Channel Shuffle |  Cutout  |
|![drawing](./images/op_jitter.gif "Jitter")|![drawing](./images/op_radial_gradient.gif "Radial Gradient")|![drawing](./images/op_channel_shuffle.gif "Channel Shuffle")|![drawing](./images/op_cutout.gif "Cutout")|
| Blend with random images | CLAHE | Contrast | Zoom |
|![drawing](./images/op_blend.gif "Blend wit random images")|![drawing](./images/op_clahe.gif "CLAHE")|![drawing](./images/op_contrast.gif "Contrast")|![drawing](./images/op_zoom.gif "Zoom")|
| Salt and pepper | Dilation | Erosion | Texture modification |
|![drawing](./images/op_salt_pepper.gif "Salt and pepper")|![drawing](./images/op_dilation.gif "Dilation")|![drawing](./images/op_erosion.gif "Erosion")|![drawing](./images/op_texture.gif "Texture modification")|
| Flashlight | Flips | Gamma | Random shadow |
|![drawing](./images/op_flashlight.gif "Flashlight")|![drawing](./images/op_flip.gif "Flips")|![drawing](./images/op_gamma.gif "Gamma")| ![drawing](./images/op_shadow.gif "Random shadow")|
| Gaussian noise | Brightness | Inversion | Rotation90 |
|![drawing](./images/op_gauss_noise.gif "Gaussian oise")|![drawing](./images/op_global_brightness.gif "Brightness")|![drawing](./images/op_inversion.gif "Inversion")|![drawing](./images/op_rotation90.gif "Rotation90")|
| Gaussian blur | Motion blur | Variable blur | Pixelize |
|![drawing](./images/op_gaussian_blur.gif "Gaussian blur")|![drawing](./images/op_motion_blur.gif "Motion blur")|![drawing](./images/op_variable_blur.gif "Variable blur")|![drawing](./images/op_pixelize.gif "Pixelization")|
| Median blur | Linear gradient | JPEG noise | Random Curve|
|![drawing](./images/op_median_blur.gif "Median blur")|![drawing](./images/op_linear_gradient.gif "Linear gradient")|![drawing](./images/op_jpeg_noise.gif "JPEG noise")|![drawing](./images/op_random_curve.gif "Random curves")|
| Elastic transformation | Optical transformation | Perspective transformation | Grid distortion  |
|![drawing](./images/op_elastic.gif "Elastic transformation")|![drawing](./images/op_optical.gif "Optical transformation")|![drawing](./images/op_perspective.gif "Perspective transformation")|![drawing](./images/op_grid.gif "Grid distortion")|
| Rotation | Scratches | Halo effect | |
|![drawing](./images/op_rotation.gif "Rotation")|![drawing](./images/op_scratches.gif "Scratches")|![drawing](./images/op_halo.gif "Halo effect")|   |


## Example usage:
```python
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
        

sample = SimpleExample().apply(aug.Sample(image, annotations, masks))


```

More: [Getting started](GETTING_STARTED.md).

## Releases

v0.1.0 - 16/07/2019
 - Initial alpha release.


## Licence
[Apache License 2.0](LICENSE)


## Contact

Project is maintained mainly by Tomasz Gilewicz ([@tgilewicz](https://github.com/tgilewicz)).
