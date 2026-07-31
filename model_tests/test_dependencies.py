from importlib.metadata import version

import numpy as np


def test_optional_medical_imaging_packages_import() -> None:
    import monai
    import torch
    import torchvision
    import torchxrayvision as xrv

    assert torch.__version__
    assert torchvision.__version__
    assert monai.__version__
    assert version("torchxrayvision") == "1.5.2"

    assert hasattr(xrv, "models")
    assert hasattr(xrv.models, "DenseNet")


def test_pytorch_can_create_cpu_image_tensor() -> None:
    import torch

    image = torch.zeros(
        (1, 1, 224, 224),
        dtype=torch.float32,
    )

    assert image.shape == (1, 1, 224, 224)
    assert image.device.type == "cpu"
    assert torch.isfinite(image).all()


def test_torchxrayvision_normalization() -> None:
    import torchxrayvision as xrv

    image = np.linspace(
        0,
        255,
        224 * 224,
        dtype=np.float32,
    ).reshape(224, 224)

    normalized = xrv.datasets.normalize(
        image,
        maxval=255,
    )

    assert normalized.shape == (224, 224)
    assert np.isfinite(normalized).all()
    assert normalized.min() >= -1024.1
    assert normalized.max() <= 1024.1
