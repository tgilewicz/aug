import csv

import cv2
import json
import matplotlib.pyplot as plt

import aug

ops = [
    aug.TextureModification(),
    aug.Scratches(),
    aug.PerspectiveDistortion(),
    aug.GridDistortion(),
    aug.OpticalDistortion(),
    aug.ElasticDistortion(),
    aug.Rotation(),
    aug.Rotation90(),
    aug.GaussNoise(),
    aug.JpegNoise(),
    aug.GlobalDarkness(),
    aug.Brightness(),
    aug.Clahe(),
    aug.Contrast(),
    aug.Gamma(),
    aug.PepperNoise(),
    aug.SaltNoise(),
    aug.ChannelShuffle(),
    aug.Inversion(),
    aug.RadialGradient(),
    aug.LinearGradient(),
    aug.Flashlight(),
    aug.CameraFlare(),
    aug.HaloEffect(),
    aug.Smudges(),
    aug.HorizontalFlip(),
    aug.VerticalFlip(),
    aug.Stretch(),
    aug.Transposition(),
    aug.BlendWithRandomImage(),
    aug.RandomRadialDirt(),
    aug.RandomShapeShadow(),
    aug.RandomCurveContour(),
    aug.Erosion(),
    aug.Dilatation(),
    aug.RandomEdge(),
    aug.Pixelize(),
    aug.GaussianBlur(),
    aug.MedianBlur(),
    aug.VariableBlur(),
    aug.MotionBlur(),
]


def show(img):
    cv2.imshow('lena.jpg', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    img = cv2.imread('lena.jpg')
    results = {}

    for op in ops:
        # show(images)
        op_result = op.time(img.copy())
        results = {**results, **op_result}
        # show(op.apply(images.copy()))

    print(json.dumps(results, indent=4))
    print('Total: {}'.format(sum(results.values())))

    plt.bar(range(len(results)), list(results.values()), align='center')
    plt.yscale('log')
    plt.xticks(range(len(results)), list(results.keys()), rotation='vertical')

    plt.show()
    with open('ops_performance.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'time'])
        w.writeheader()
        for k, v in results.items():
            w.writerow({'name': k, 'time': v})


if __name__ == "__main__":
    main()
