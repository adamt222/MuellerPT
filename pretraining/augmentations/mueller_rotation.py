from __future__ import annotations

import math
import numbers
import random

import torch
import torchvision.transforms.functional as F


class RandomMuellerRotationCustom(object):
    """Rotate the Mueller matrix frame by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
        p (float): probability threshold with which image should be rotated or left untreated instead.
        any (bool): If false, only angles [90, 180, 270] will be chosen. Otherwise, a random angle
            within the boundaries will be generated (default).

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(
        self,
        degrees,
        resample=False,
        expand=False,
        center=None,
        pad_rotate=None,
        fill=0,
        p=0.5,
        any=True,
    ):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.pad_rotate = pad_rotate
        self.fill = fill
        self.p = p
        self.any = any

    def get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """

        angle = random.uniform(degrees[0], degrees[1]) if self.any else random.choice([90, 180, 270])

        return angle

    @staticmethod
    def get_rmat(deg):
        """Get rotation matrix for Mueller matrix.

        Returns:
            sequence: rotation matrix for Mueller matrix.
        """
        theta = deg / 180 * math.pi
        rmat = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, math.cos(2 * theta), -math.sin(2 * theta), 0],
                [0, math.sin(2 * theta), +math.cos(2 * theta), 0],
                [0, 0, 0, 1],
            ]
        )

        return rmat

    def __call__(self, img, label=None, angle=None, center=None, *args, **kwargs):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        if random.random() < self.p:
            # spatial transformation
            angle = self.get_params(self.degrees) if angle is None else angle
            return self.apply_rotation(img, angle=angle, center=center, label=label)
        else:
            if label is not None:
                return img, label
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "(degrees={0}".format(self.degrees)
        format_string += ", resample={0}".format(self.resample)
        format_string += ", expand={0}".format(self.expand)
        if self.center is not None:
            format_string += ", center={0}".format(self.center)
        format_string += ")"
        return format_string

    def apply_rotation(self, img, angle, center=None, label=None):
        """
        Apply rotation with a provided angle (no randomness).
        """
        self.center = self.center if center is None else center
        if self.pad_rotate is None:
            fill16 = torch.eye(4).flatten().tolist()
            rotated_img = F.rotate(
                img,
                angle,
                interpolation=self.resample,
                expand=self.expand,
                center=self.center,
                fill=fill16,
            )
        else:
            rotated_img = self.pad_rotate(
                img, angle, interpolation=self.resample, expand=self.expand, center=self.center
            )
        rotated_img = rotated_img.moveaxis(0, 2)
        # mueller matrix transformation
        T = self.get_rmat(-1 * angle)
        rotated_img = T @ rotated_img.view(*rotated_img.shape[:-1], 4, 4) @ T.transpose(-2, -1)
        rotated_img = rotated_img.flatten(-2, -1).moveaxis(2, 0)
        if label is not None:
            rotated_label = F.rotate(label, angle, self.resample, self.expand, self.center, self.fill)
            return rotated_img, rotated_label
        return rotated_img


class RandomMuellerRotationExpandCrop(RandomMuellerRotationCustom):
    """Rotate with expand=True and center-crop back to original size."""

    def apply_rotation(self, img, angle, center=None, label=None):
        self.center = self.center if center is None else center
        orig_h, orig_w = img.shape[-2], img.shape[-1]
        if self.pad_rotate is None:
            rotated_img = F.rotate(
                img,
                angle,
                interpolation=self.resample,
                expand=True,
                center=self.center,
                fill=self.fill,
            )
        else:
            rotated_img = self.pad_rotate(
                img, angle, interpolation=self.resample, expand=True, center=self.center
            )
        rotated_img = F.center_crop(rotated_img, [orig_h, orig_w])
        rotated_img = rotated_img.moveaxis(0, 2)
        T = self.get_rmat(-1 * angle)
        rotated_img = T @ rotated_img.view(*rotated_img.shape[:-1], 4, 4) @ T.transpose(-2, -1)
        rotated_img = rotated_img.flatten(-2, -1).moveaxis(2, 0)
        if label is not None:
            rotated_label = F.rotate(label, angle, self.resample, True, self.center, self.fill)
            rotated_label = F.center_crop(rotated_label, [orig_h, orig_w])
            return rotated_img, rotated_label
        return rotated_img
