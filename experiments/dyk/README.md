# Dyck-k Experiments

Experiments for training and evaluating SplineTransformer models on Dyck-k formal languages.
These experiments investigate how spline geometry features relate to aleatoric and epistemic uncertainty.

## Overview

Dyck-k languages consist of properly nested bracket sequences with k different bracket types.
For example, Dyck-2 uses `()` and `[]`, while Dyck-4 adds `{}` and `<>`.

The experiments in this directory:
1. Train transformer models on next-token prediction for Dyck sequences
2. Evaluate prediction accuracy and generalization
3. Analyze spline geometry features from the gated MLPs
4. Test whether spline features correlate with uncertainty

## Files

### Training

| File | Description |
|------|-------------|
| `train.py` | Train SplineTransformer on Dyck-k sequences with autoregressive LM objective |

### Evaluation

| File | Description |
|------|-------------|
| `evaluate.py` | Basic evaluation: test final token prediction accuracy |
| `evaluate_uncertainty.py` | Analyze spline features vs aleatoric uncertainty (single k) |
| `evaluate_mixed.py` | Mixed Dyck-2 to Dyck-k: correlate features with ambiguity levels |
| `evaluate_epistemic.py` | Compare ID vs OOD feature distributions |
| `evaluate_intrinsic_dim.py` | Replicate ID vs correctness correlation (arXiv:2407.02678) |

### Analysis

| File | Description |
|------|-------------|
| `analyze_splines.py` | Extract and analyze the 7 spline features from the paper |
| `classify_ood.py` | Train binary classifier for OOD detection using spline features |
| `regress_aleatoric.py` | Regressor to predict n_valid from spline features |

## Quick Start

```bash
# 1. Train a mixed model (Dyck-2 to Dyck-8)
python -m experiments.dyk.train --k=8 --mixed --steps=5000 --save_path="models/dyck_mixed.pt"

# 2. Evaluate aleatoric uncertainty
python -m experiments.dyk.evaluate_mixed --checkpoint=models/dyck_mixed.pt --max_k=8 --plot

# 3. Train for epistemic uncertainty (hold out Dyck-7,8)
python -m experiments.dyk.train --k=8 --mixed --max_k_train=6 --steps=5000 --save_path="models/dyck_2to6.pt"

# 4. Evaluate epistemic uncertainty
python -m experiments.dyk.evaluate_epistemic --checkpoint=models/dyck_2to6.pt --k_train=6 --k_test=8 --plot

# 5. Binary classification experiment
python -m experiments.dyk.classify_ood --checkpoint=models/dyck_2to6.pt --k_train=6 --k_test=8 --plot

# 6. Regression experiment
python -m experiments.dyk.regress_aleatoric --checkpoint=models/dyck_mixed.pt --max_k=8 --plot
```

## Key Concepts

### Aleatoric Uncertainty
Inherent ambiguity in the prediction task. In Dyck-k, this is the number of valid next tokens:
- Empty stack: k valid options (any open bracket)
- Non-empty stack: k+1 valid options (any open bracket + matching close)

### Epistemic Uncertainty
Model uncertainty due to lack of knowledge. Tested by evaluating on OOD data (bracket types not seen during training).

### Spline Features (7 features from the paper)
1. `feature_1`: Global sign density (mean fraction of positive activations)
2. `feature_2`: Min sign density across tokens
3. `feature_3`: Max sign density across tokens
4. `feature_4`: Std of sign density
5. `feature_5`: Global closest distance to hyperplane
6. `feature_6`: Mean distance to hyperplanes
7. `feature_7`: Std of distances

### Additional Features
- `q10`: 10th percentile distance (from `spline_features_lasttok_softmin`)
- `softmin`: Smooth minimum distance
- `sign_density`: Fraction of active neurons at last token
- `entropy`: Prediction entropy (-sum(p log p))

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- tqdm
- scipy
- scikit-learn (for classify_ood.py and regress_aleatoric.py)

## Related Files

- `data/dyk/dyk.py`: DyckPCFG class for sequence generation
- `architectures/transformers.py`: SplineTransformer model
- `architectures/utils.py`: Spline feature extraction functions
