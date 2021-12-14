import torch
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, 
                 n_holes, 
                 length, 
                 prob):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.random() < self.prob:
            return img
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class DualCutout(object):
    def __init__(self, 
                 n_holes, 
                 length, 
                 prob):
        self.cutout = Cutout(n_holes, length, prob)

    def __call__(self, image) :
        return np.hstack([self.cutout(image), self.cutout(image)])

class RandomErasing(object):
    def __init__(self, 
                 prob, 
                 max_attempt, 
                 area_ratio_range, 
                 min_aspect_ratio):
        self.p = prob
        self.max_attempt = max_attempt
        self.sl, self.sh = area_ratio_range
        self.rl = min_aspect_ratio
        self.rh = 1. / min_aspect_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image

"""Base augmentations operators."""
def augmix(cfg, image, preprocess):
    
    aug_list = augmentations
    ws = np.float32(
        np.random.dirichlet([cfg.aug.prob_coeff] * cfg.aug.mixture_width))
    m = np.float32(np.random.beta(cfg.aug.prob_coeff, cfg.aug.prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(cfg.aug.mixture_width):
        image_aug = image.copy()
        depth = cfg.aug.mixture_depth if cfg.aug.mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, cfg.aug.severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 128


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

# try:
#     import albumentations
#     from albumentations import Compose
# except ImportError:
#     albumentations = None
#     Compose = None

# class Albu(object):
#     """Albumentation augmentation.
#     Adds custom transformations from Albumentations library.
#     Please, visit `https://albumentations.readthedocs.io`
#     to get more information.
#     An example of ``transforms`` is as followed:
#     .. code-block::
#         [
#             dict(
#                 type='ShiftScaleRotate',
#                 shift_limit=0.0625,
#                 scale_limit=0.0,
#                 rotate_limit=0,
#                 interpolation=1,
#                 p=0.5),
#             dict(
#                 type='RandomBrightnessContrast',
#                 brightness_limit=[0.1, 0.3],
#                 contrast_limit=[0.1, 0.3],
#                 p=0.2),
#             dict(type='ChannelShuffle', p=0.1),
#             dict(
#                 type='OneOf',
#                 transforms=[
#                     dict(type='Blur', blur_limit=3, p=1.0),
#                     dict(type='MedianBlur', blur_limit=3, p=1.0)
#                 ],
#                 p=0.1),
#         ]
#     Args:
#         transforms (list[dict]): A list of albu transformations
#         keymap (dict): Contains {'input key':'albumentation-style key'}
#     """

#     def __init__(self, transforms, keymap=None, update_pad_shape=False):
#         if Compose is None:
#             raise RuntimeError('albumentations is not installed')

#         self.transforms = transforms
#         self.filter_lost_elements = False
#         self.update_pad_shape = update_pad_shape

#         self.aug = Compose([self.albu_builder(t) for t in self.transforms])

#         if not keymap:
#             self.keymap_to_albu = {
#                 'img': 'image',
#             }
#         else:
#             self.keymap_to_albu = keymap
#         self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

#     def albu_builder(self, cfg):
#         """Import a module from albumentations.
#         It inherits some of :func:`build_from_cfg` logic.
#         Args:
#             cfg (dict): Config dict. It should at least contain the key "type".
#         Returns:
#             obj: The constructed object.
#         """

#         assert isinstance(cfg, dict) and 'type' in cfg
#         args = cfg.copy()

#         obj_type = args.pop('type')
#         if mmcv.is_str(obj_type):
#             if albumentations is None:
#                 raise RuntimeError('albumentations is not installed')
#             obj_cls = getattr(albumentations, obj_type)
#         elif inspect.isclass(obj_type):
#             obj_cls = obj_type
#         else:
#             raise TypeError(
#                 f'type must be a str or valid type, but got {type(obj_type)}')

#         if 'transforms' in args:
#             args['transforms'] = [
#                 self.albu_builder(transform)
#                 for transform in args['transforms']
#             ]

#         return obj_cls(**args)

#     @staticmethod
#     def mapper(d, keymap):
#         """Dictionary mapper. Renames keys according to keymap provided.
#         Args:
#             d (dict): old dict
#             keymap (dict): {'old_key':'new_key'}
#         Returns:
#             dict: new dict.
#         """

#         updated_dict = {}
#         for k, v in zip(d.keys(), d.values()):
#             new_k = keymap.get(k, k)
#             updated_dict[new_k] = d[k]
#         return updated_dict

#     def __call__(self, results):
#         # dict to albumentations format
#         results = self.mapper(results, self.keymap_to_albu)

#         results = self.aug(**results)

#         if 'gt_labels' in results:
#             if isinstance(results['gt_labels'], list):
#                 results['gt_labels'] = np.array(results['gt_labels'])
#             results['gt_labels'] = results['gt_labels'].astype(np.int64)

#         # back to the original format
#         results = self.mapper(results, self.keymap_back)

#         # update final shape
#         if self.update_pad_shape:
#             results['pad_shape'] = results['img'].shape

#         return results