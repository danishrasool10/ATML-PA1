import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def domain_index(domain: str) -> int:
    return PACS_DOMAINS.index(domain)

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def _find_child_dir(parent: Path, wanted: str) -> Optional[Path]:
    for child in sorted(parent.iterdir()):
        if child.is_dir() and _norm(child.name) == _norm(wanted):
            return child
    return None

def _has_all_domains(path: Path) -> bool:
    return all(_find_child_dir(path, d) is not None for d in PACS_DOMAINS)

def resolve_pacs_root(root) -> Path:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"PACS root does not exist: {root}")
    if _has_all_domains(root):
        return root
    
    level1 = [p for p in sorted(root.iterdir()) if p.is_dir()]
    level2 = [q for p in level1 for q in sorted(p.iterdir()) if q.is_dir()]
    for cand in level1 + level2:
        if _has_all_domains(cand):
            return cand
    raise FileNotFoundError(f"Could not find domain folders under {root}")

def list_domain_samples(root, domain: str) -> List[Tuple[str, int]]:
    root = Path(root)
    domain_dir = _find_child_dir(root, domain)
    if domain_dir is None:
        raise FileNotFoundError(f"Domain folder '{domain}' not found in {root}")
        
    samples = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = _find_child_dir(domain_dir, cls)
        if cls_dir is None:
            raise FileNotFoundError(f"Class folder '{cls}' not found in {domain_dir}")
            
        files = sorted(p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
        samples.extend((p.relative_to(root).as_posix(), label) for p in files)
    return samples

def imagenet_normalization() -> Tuple[List[float], List[float]]:
    t = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return list(t.mean), list(t.std)

def build_transforms(train: bool, image_size: int = 256, crop_size: int = 224) -> Callable:
    mean, std = imagenet_normalization()
    ops = [transforms.Resize((image_size, image_size))]
    ops += [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()] if train else [transforms.CenterCrop(crop_size)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)

class PACSDataset(Dataset):
    """Returns (image, label, domain_id). With use_labels=False, the label is always -1."""
    def __init__(self, root, samples, domain: str, transform: Callable, use_labels: bool = True):
        self.root = Path(root)
        self.samples = [(str(p), int(y)) for p, y in samples]
        self.domain = domain
        self.domain_id = domain_index(domain)
        self.transform = transform
        self.use_labels = use_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel, label = self.samples[index]
        with Image.open(self.root / rel) as img:
            image = self.transform(img.convert("RGB"))
        return image, (label if self.use_labels else -1), self.domain_id