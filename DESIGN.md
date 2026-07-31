# RadiantGuard AI — Design Record

## Product thesis

Medical-imaging AI should not be evaluated only by whether it produces a prediction. A responsible system should also expose its uncertainty, document its data and preprocessing, help users evaluate whether explanations align with model behavior, and preserve qualified human oversight.

## Foundation decision: do not ship a fake predictor

The first version does not simulate medical findings from uploaded images. A polished but fabricated prediction could be mistaken for a real analysis and would undermine the project's safety mission. The model workspace therefore remains visibly disabled until a selected public research model is integrated with documented weights, preprocessing, validation, and limitations.

## Architecture

The foundation is separated into three concerns:

- `src/imaging.py`: safe file parsing, DICOM pixel extraction, normalization, and non-clinical technical checks.
- `src/safety.py`: rules-based review of AI-generated report wording.
- `src/ui.py`: interface theme and reusable presentation components.
- `app.py`: application composition and user flow.

The model layer will later live under `src/models/` and expose a stable interface rather than being embedded directly in Streamlit UI code.

## Safety-focused UI decisions

1. The medical-use boundary appears before the workspace.
2. Upload is disabled until the user acknowledges the prototype and privacy boundary.
3. The interface does not display patient name, patient ID, accession number, date of birth, or institution fields.
4. Technical pixel checks are explicitly distinguished from clinical image-quality assessment.
5. The model area states that no inference is enabled rather than showing sample findings beside a user's image.

## Intended model interface

```python
class ImagingResearchModel(Protocol):
    model_id: str
    version: str

    def preprocess(self, image: np.ndarray) -> ModelInput: ...
    def predict(self, model_input: ModelInput) -> PredictionBundle: ...
    def explain(self, model_input: ModelInput, target: str) -> ExplanationMap: ...
```

Every `PredictionBundle` should include model identity, preprocessing version, labels, raw scores, calibrated probabilities where available, and warnings.

## Trust score warning

A future "trust score" must not collapse complex evidence into an unexplained number. If implemented, it should be decomposable into visible factors such as image compatibility, calibration, model agreement, explanation stability, and output-language safety. It must never imply clinical safety or FDA authorization.
