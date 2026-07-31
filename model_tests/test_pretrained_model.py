import torch
import torchxrayvision as xrv


def test_pretrained_chest_model_loads_and_runs_on_cpu() -> None:
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all",
    )

    model.eval()
    model.to("cpu")

    image = torch.zeros(
        (1, 1, 224, 224),
        dtype=torch.float32,
        device="cpu",
    )

    with torch.inference_mode():
        output = model(image)

    assert output.shape == (1, 18)
    assert len(model.pathologies) == 18
    assert torch.isfinite(output).all()

    assert "Pneumonia" in model.pathologies
    assert "Pneumothorax" in model.pathologies
    assert "Cardiomegaly" in model.pathologies


def test_pretrained_model_output_matches_pathology_names() -> None:
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all",
    )

    model.eval()

    image = torch.zeros(
        (1, 1, 224, 224),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = model(image)[0]

    results = dict(
        zip(
            model.pathologies,
            output.tolist(),
            strict=True,
        )
    )

    assert len(results) == 18
    assert all(
        isinstance(label, str)
        for label in results
    )
    assert all(
        isinstance(score, float)
        for score in results.values()
    )
