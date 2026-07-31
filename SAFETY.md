# Safety and Responsible Use

## Status

RadiantGuard AI is an educational research prototype. It is not clinically validated, not an FDA-authorized medical device, and not intended for patient care.

## Prohibited uses

Do not use RadiantGuard AI to:

- Diagnose, exclude, confirm, triage, or treat a medical condition
- Delay or replace qualified physician or radiologist review
- Make emergency, medication, surgical, or follow-up decisions
- Process identifiable patient information in an unapproved environment
- Represent prototype output as a radiology report

## Required interpretation

All prototype information may be incomplete, incorrect, biased, or inapplicable to a particular image. Any medical image must be reviewed in its complete clinical context by a qualified physician or radiologist.

## Data policy

Development should use only public, appropriately licensed, synthetic, or properly de-identified datasets. Dataset documentation must record:

- Source and license
- Intended and prohibited uses
- Label-generation method
- Patient population and setting
- Known exclusions and missingness
- Potential demographic and acquisition biases
- Train, validation, and test split design

## Model-release gate

A research model should not be enabled in the public interface until the repository includes:

1. Model identity, version, source, license, and checksum
2. Exact preprocessing and label definitions
3. Evaluation on a held-out dataset
4. Calibration assessment
5. Known limitations and failure modes
6. Human-readable output restrictions
7. A model card
8. Clear separation between research output and diagnosis

## Incident handling

Any issue that could cause a user to mistake the prototype for clinical advice should be treated as a high-priority defect. The safest response may be to disable the affected feature until its presentation and limitations are corrected.
