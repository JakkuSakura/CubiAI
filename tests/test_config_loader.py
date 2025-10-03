from cubiai.config.loader import load_config


def test_config_loader_default() -> None:
    cfg = load_config()
    assert cfg.name == "anime-default"
    assert cfg.segmentation.backend == "sam-hq-local"
    assert cfg.segmentation.sam_hq_local_model_id == "syscv-community/sam-hq-vit-base"
    assert cfg.export.psd.enabled is True
    assert cfg.export.png.enabled is True
    assert cfg.rigging.enabled is False
    assert cfg.export.live2d.enabled is False
