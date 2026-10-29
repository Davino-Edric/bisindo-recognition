# BISINDO Sign Language Recognition (Training Repo)

Recognizes the 26 letters of the BISINDO (Indonesian Sign Language) alphabet from hand landmarks. MediaPipe Hands extracts two-hand landmarks from an image, and a PyTorch MLP classifies them.

- **Live demo:** https://huggingface.co/spaces/vino-edric/bisindo-space
- **This repo:** data pipeline, training, evaluation, MLflow experiments.
- **Deployment repo:** inference code and model artifact only (the Space).

> Status: static alphabet recognition is built and deployed. Dynamic signs (LSTM) are planned.

---

## Results

| Split | Accuracy | Notes |
|---|---|---|
| Validation (15%) | ~95% | Used for model selection |
| Test (15%) | **TBD** | Held out, not yet evaluated |
| Live webcam | **TBD** | Manual test on the deployed Space |

<!-- TODO: fill the TBD rows after running the test split. Do not publish the validation number as the headline. -->

### Per-class accuracy

TBD. Generate from the test split and paste here.

### Confusion matrix

![Confusion matrix](reports/confusion_matrix.png)

<!-- TODO: save the test-split confusion matrix to this path, then name the 2-3 most confused letter pairs below. -->

---

## Why MediaPipe landmarks

The first prototype was built on Teachable Machine and classified raw pixels. It overfit to the small set of hands it was trained on.

This version feeds the classifier hand geometry instead of pixels:

- MediaPipe Hands returns 21 3D landmarks per hand.
- Background, skin tone, and lighting no longer enter the model directly.
- The input shrinks from an image to a small numeric vector, so a simple MLP is enough.

Trade-off: the model is only as good as MediaPipe's hand detection. If the hand is not detected, there is no prediction.

---

## Data

| Item | Value |
|---|---|
| Sources | 3 merged public datasets (achmadnoer, agungmrf, Zappie733) |
| Usable samples | 9,599 |
| Classes | 26 (A-Z) |
| Raw layout | `data/raw/{letter}/` |
| Split | 70 / 15 / 15, stratified by class |

### Feature layout (126 features per sample)

| Index | Content |
|---|---|
| 0-62 | Right hand: 21 landmarks x (x, y, z) |
| 63-125 | Left hand: 21 landmarks x (x, y, z) |

A missing hand is filled with zeros.

<!-- TODO: add 1-2 lines on normalization (e.g. relative to wrist, scaled) exactly as your code does it. -->

---

## Model

PyTorch MLP: `126 -> 256 -> 128 -> 26`

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Dropout | 0.3 |
| Loss | Cross-entropy |
| Tracking | MLflow |

The trained weights are saved with `torch.save(model.state_dict(), ...)` and loaded into the same architecture at inference time.

<!-- TODO: add learning rate, batch size, epochs, and early stopping if used. -->

---

## Experiment tracking

Each run logs parameters, per-epoch metrics, and the model artifact to MLflow.

```bash
mlflow ui
```

<!-- TODO: add a screenshot of the run comparison page. -->

---

## Reproduce

```bash
pip install -r requirements.txt
# 1. place the raw images under data/raw/{letter}/
# 2. extract landmarks  -> <your extraction script>
# 3. train              -> <your training script>
# 4. evaluate           -> <your evaluation script>
```

<!-- TODO: replace the placeholders with the real script names and confirm requirements.txt exists. -->

---

## Limitations

- **Static alphabet only.** Letters that need motion are not covered yet.
- **Dataset diversity.** Three public datasets, so the number of distinct signers is limited and unverified.
- **Two-hand ordering.** Features depend on the right/left assignment from MediaPipe. A wrong assignment changes the input.
- **Webcam conditions.** Lighting, camera angle, and distance were not systematically tested.
- **Validation vs test.** Reported validation accuracy may be optimistic until the test split is evaluated.

---

## Roadmap

- [ ] Evaluate on the held-out test split and publish per-class results
- [ ] Check for duplicate frames across train/val/test
- [ ] LSTM on landmark sequences for dynamic signs
- [ ] Systematic live-webcam evaluation

---

## Author

Davino Edric Faturrahman, Data Science (Sains Data Terapan), PENS.