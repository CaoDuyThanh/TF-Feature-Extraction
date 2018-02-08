import imgaug as ia
from imgaug import augmenters as iaa
import numpy

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
data_aug_manager = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent  = (-0.5, 0.5),
            pad_mode = ia.ALL,
            pad_cval = (0, 255)
        )),
        sometimes(iaa.Affine(
            scale             = {"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            rotate            = (-45, 45), # rotate by -45 to +45 degrees
            shear             = (-16, 16), # shear by -16 to +16 degrees
        ))
    ],
    random_order = True
)