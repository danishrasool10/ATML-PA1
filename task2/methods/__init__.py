from task2.methods.source_only import SourceOnly
from task2.methods.dan import DAN, multi_kernel_mmd
from task2.methods.dann import DANN
from task2.methods.cdan import CDAN
from task2.models.domain_discriminator import DomainDiscriminator


def build_method(cfg, feature_dim: int, num_classes: int):
    name = cfg["method"]
    if name == "source_only":
        return SourceOnly()
    if name == "dan":
        return DAN(lambda_mmd=cfg.get("lambda_mmd", 1.0))
    if name == "dann":
        disc = DomainDiscriminator(in_dim=feature_dim)
        return DANN(
            disc,
            lambda_domain=cfg.get("lambda_domain", 1.0),
            grl_max=cfg.get("grl_max", 1.0),
        )
    if name == "cdan":
        disc = DomainDiscriminator(in_dim=feature_dim * num_classes)
        return CDAN(
            disc,
            lambda_domain=cfg.get("lambda_domain", 1.0),
            grl_max=cfg.get("grl_max", 1.0),
        )
    raise ValueError(f"Unknown method: {name}")


__all__ = ["SourceOnly", "DAN", "DANN", "CDAN", "multi_kernel_mmd", "build_method"]
