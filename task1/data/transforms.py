import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMAGE_SIZE = 224
HUE_SHIFT = 0.3
TRANSLATIONS_PX = (0, 8, 16, 32)
DIRECTIONS = ("up", "down", "left", "right")
PATCH_GRID = 4


def base_transform(image_size=IMAGE_SIZE):
    return T.Compose([
        T.Resize(int(image_size * 1.15)),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])


def to_grayscale_3ch(img):
    return TF.rgb_to_grayscale(img, num_output_channels=3)


def hue_rotate(img, hue_shift=HUE_SHIFT):
    return TF.adjust_hue(img.clamp(0, 1), hue_shift)


def _reflect_pad_and_crop(img, dx, dy):
    pad = max(abs(dx), abs(dy))
    if pad == 0:
        return img.clone()
    
    c, h, w = img.shape
    padded = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    
    top = pad - dy
    left = pad - dx
    return padded[:, top:top + h, left:left + w]


def translate_image(img, pixels, direction):
    assert direction in DIRECTIONS
    if pixels == 0:
        return img.clone()
        
    dx, dy = {
        "left": (-pixels, 0), "right": (pixels, 0),
        "up": (0, -pixels), "down": (0, pixels),
    }[direction]
    
    return _reflect_pad_and_crop(img, dx, dy)


def four_direction_translations(img, pixels):
    return {d: translate_image(img, pixels, d) for d in DIRECTIONS}


def _patch_permutation(image_id, grid=PATCH_GRID, seed=6304):
    n = grid * grid
    rng = np.random.RandomState((seed + image_id * 7919) % (2 ** 31 - 1))
    perm = np.arange(n)
    
    for _ in range(50):
        rng.shuffle(perm)
        if not np.array_equal(perm, np.arange(n)):
            return perm
            
    perm[0], perm[1] = perm[1], perm[0] 
    return perm


def patch_shuffle(img, image_id, grid=PATCH_GRID, seed=6304):
    c, h, w = img.shape
    assert h % grid == 0 and w % grid == 0
    
    ph, pw = h // grid, w // grid
    
    patches = img.reshape(c, grid, ph, grid, pw).permute(1, 3, 0, 2, 4).reshape(grid * grid, c, ph, pw)
    perm = _patch_permutation(image_id, grid, seed)
    shuffled = patches[torch.as_tensor(perm, dtype=torch.long)]
    
    out = shuffled.reshape(grid, grid, c, ph, pw).permute(2, 0, 3, 1, 4).reshape(c, h, w)
    return out


def set_all_seeds(seed=6304):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)