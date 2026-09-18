import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

try:
    import open_clip
except ImportError as e:
    raise ImportError("pip install open_clip_torch") from e

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


class ResNet50Backbone(nn.Module):
    feature_dim = 2048
    name = "resnet50"

    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        self.model = m

    @torch.no_grad()
    def extract_features(self, images_01: torch.Tensor) -> torch.Tensor:
        x = _normalize(images_01, IMAGENET_MEAN, IMAGENET_STD)
        return self.model(x)


class ViTB16Backbone(nn.Module):
    feature_dim = 768
    name = "vit_b_16"

    def __init__(self):
        super().__init__()
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        self.model = m

    @torch.no_grad()
    def extract_features(self, images_01: torch.Tensor) -> torch.Tensor:
        x = _normalize(images_01, IMAGENET_MEAN, IMAGENET_STD)
        return self.model(x)


class CLIPViTB32Backbone(nn.Module):
    feature_dim = 512
    name = "clip_vit_b_32"

    def __init__(self, device: str = "cpu"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logit_scale = model.logit_scale.exp().item()

    @torch.no_grad()
    def extract_features(self, images_01: torch.Tensor) -> torch.Tensor:
        x = _normalize(images_01, CLIP_MEAN, CLIP_STD)
        feats = self.model.encode_image(x)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_class_prompts(self, class_names, prompt_template: str = "a photo of a {}.") -> torch.Tensor:
        prompts = [prompt_template.format(c) for c in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        text_feats = self.model.encode_text(tokens)
        return text_feats / text_feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def zero_shot_logits(self, images_01: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        img_feats = self.extract_features(images_01)
        return self.logit_scale * img_feats @ text_features.t()


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


def build_backbones(device: str = "cpu") -> dict:
    return {
        "resnet50": ResNet50Backbone().to(device),
        "vit_b_16": ViTB16Backbone().to(device),
        "clip_vit_b_32": CLIPViTB32Backbone(device=device).to(device),
    }