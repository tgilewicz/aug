import setuptools
import aug

with open("README.md", "r") as fh:
    long_description = fh.read().split('## Installation')[0]


setuptools.setup(
    name="aug",
    version=aug.__version__,
    author="Tomasz Gilewicz",
    author_email="tomasz.gilewicz1@gmail.com",
    description="Augmentation library based on OpenCV.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cta-ai/aug",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
    keywords="data augmentation artificial synthesis machine learning",
    install_requires=[
        "imageio==2.5.0",
        "matplotlib==3.1.0",
        "numpy==1.16.4",
        "opencv-python==4.1.0.25",
        "Pillow==6.0.0",
        "requests==2.22.0",
        "scipy==1.3.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
