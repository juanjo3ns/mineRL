import numpy as np
import torch


def numpify(func):
    """Wrapper so that the augmentation function always works on a numpy
    array, but if the input `imgs` is a torch tensor, a torch tensor will be
    returned. Assumes first input and first output of the function is the
    images array/tensor, and only operates on that."""
    def numpified_aug(imgs, *args, **kwargs):
        _numpify = isinstance(imgs, torch.Tensor)
        if _numpify:
            imgs = imgs.numpy()
        ret = func(imgs, *args, **kwargs)
        if _numpify:
            if isinstance(ret, tuple):
                # Assume first is the augmented images.
                ret = (torch.from_numpy(ret[0]), *ret[1:])
            else:
                ret = torch.from_numpy(ret)
        return ret
    return numpified_aug

@numpify
def random_shift(imgs, pad=1, prob=1.):
    t = b = c = 1
    shape_len = len(imgs.shape)
    if shape_len == 2:  # Could also make all this logic into a wrapper.
        h, w = imgs.shape
    elif shape_len == 3:
        c, h, w = imgs.shape
    elif shape_len == 4:
        b, c, h, w = imgs.shape
    elif shape_len == 5:
        t, b, c, h, w = imgs.shape  # Apply same crop to all T
        imgs = imgs.transpose(1, 0, 2, 3, 4)
        _c = c
        c = t * c
        # imgs = imgs.reshape(b, t * c, h, w)
    imgs = imgs.reshape(b, c, h, w)

    crop_h = h
    crop_w = w

    padded = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    b, c, h, w = padded.shape

    h_max = h - crop_h + 1
    w_max = w - crop_w + 1
    h1s = np.random.randint(0, h_max, b)
    w1s = np.random.randint(0, w_max, b)
    if prob < 1.:
        which_no_crop = np.random.rand(b) > prob
        h1s[which_no_crop] = pad
        w1s[which_no_crop] = pad

    shifted = np.zeros_like(imgs)
    for i, (pad_img, h1, w1) in enumerate(zip(padded, h1s, w1s)):
        shifted[i] = pad_img[:, h1:h1 + crop_h, w1:w1 + crop_w]

    if shape_len == 2:
        shifted = shifted.reshape(crop_h, crop_w)
    elif shape_len == 3:
        shifted = shifted.reshape(c, crop_h, crop_w)
    elif shape_len == 5:
        shifted = shifted.reshape(b, t, _c, crop_h, crop_w)
        shifted = shifted.transpose(1, 0, 2, 3, 4)

    return shifted
