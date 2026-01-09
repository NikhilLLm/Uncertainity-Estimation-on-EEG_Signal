
# EEG-based Motor Imagery Classification with Aleatoric Uncertainty Estimation

This project focuses on estimating **aleatoric uncertainty** in EEG-based motor imagery classification using the **BCI Competition IV 2a** dataset. We apply **Test-Time Data Augmentation (TTDA)** techniques to capture the uncertainty caused by inherent data noise and variability.

---

## 📌 Table of Contents

- [Dataset](#dataset)
- [Aleatoric Uncertainty Estimation](#aleatoric-uncertainty-estimation)
  - [1. EMD-TTA](#1-emd-tta-empirical-mode-decomposition)
  - [2. ICA + FGSM TTA](#2-ica--fgsm-based-tta)
- [Metrics Used](#metrics-used)
- [Preliminary Results](#preliminary-results)

---

## Dataset

### BCI Competition IV 2a

- **Subjects**: 9
- **Channels**: 22 EEG + 3 EOG
- **Sampling Rate**: 250 Hz
- **Tasks**: Left Hand, Right Hand, Feet, Tongue
- **Total Trials**: 576 per subject (288 train + 288 eval)
- **Trial Duration**: 7.5 seconds (motor imagery from 3s to 6s)

---

## Aleatoric Uncertainty Estimation

We explore **two methods** to model data-based uncertainty using TTDA.

---

### 1. EMD-TTA (Empirical Mode Decomposition)

We apply EMD on EEG trials to generate **artificial EEG frames** for Test-Time Data Augmentation.

- Each EEG trial is decomposed into **Intrinsic Mode Functions (IMFs)** using the PyEMD library.
- Selectively recombine subsets of IMFs to generate **new artificial EEG frames**.
- These frames simulate signal variations encountered in real data.

#### Experimental Setup:
- Number of Artificial Frames: 10, 20, 30, 40, 50
- IMFs used: 3, 5, 7, 9, 12
- Metrics: Mean Accuracy, Accuracy Variance, Mean Kappa, Kappa Variance

📊 _Plot: Accuracy vs Number of Frames (example)_  
![EMD Results](results_png/emd_results.png)

---
### 2. ICA + FGSM-based TTA (My Implementation)

We generate **adversarial yet physiologically-constrained** EEG variations using Independent Component Analysis (ICA) combined with Fast Gradient Sign Method (FGSM).

> **Implementation Note**: This pipeline was independently designed and implemented by me, synthesizing adversarial augmentation techniques with neuroscience-informed constraints to ensure physiological validity.

---

#### Method Pipeline:

##### 1. **ICA Decomposition**
   - Decompose each EEG trial (22 channels × time samples) into **22 independent components (ICs)** using FastICA
   - **Component Selection Criteria**:
     - **Spectral Power**: Retain ICs with dominant power in task-relevant frequency bands:
       - 7–13 Hz (mu rhythm) for left/right hand imagery
       - 12–32 Hz (beta rhythm) for feet/tongue imagery
     - **Topographic Relevance**: Prioritize ICs spatially correlated with motor cortex channels (C3, C4, Cz)
   - Rationale: Isolates brain signals from artifacts (eye blinks, muscle noise)

##### 2. **FGSM Perturbation with Physiological Constraints**
   - Compute gradients on selected ICs with respect to classification loss
   - Apply epsilon-scaled perturbations using FGSM: `IC_perturbed = IC + ε × sign(∇_IC Loss)`
   - **Enforce Physiological Validity**:
     - Apply bandpass filter (7–32 Hz) to perturbed ICs to remove unrealistic frequency components
     - Add small Gaussian noise (σ = 0.01) to simulate natural EEG variability
   - Purpose: Generate adversarial examples that remain within the manifold of real EEG signals

##### 3. **Reconstruction and Ensemble Prediction**
   - Apply inverse ICA transformation to reconstruct full 22-channel augmented EEG
   - Generate multiple augmented versions per trial (varying epsilon and number of augmentations)
   - Aggregate predictions using **cosine similarity** across top-3 softmax probability vectors
   - Output: Ensemble prediction with uncertainty estimate (softmax entropy)

---

#### Experimental Setup:

| Parameter | Values Tested |
|-----------|---------------|
| **Epsilon (ε)** | 0.01, 0.1, 0.2, 0.3, 0.4 |
| **Augmentations per Trial** | 40, 80, 120, 160, 200 |
| **Model** | ATCNet (state-of-the-art for motor imagery) |
| **Evaluation** | 10 independent runs |
| **Metrics** | Accuracy, Cohen's Kappa, Softmax Entropy |

---

#### Results:

📊 **Accuracy and Uncertainty Analysis**

<p align="center">
  <img src="results_png/ica_fsgm_results.png" alt="ICA+FGSM results showing accuracy stability and uncertainty quantification" width="100%"/>
</p>

**Key Observations:**

1. **Accuracy Stability (Top Panels)**:
   - Mean accuracy remains consistent at ~84-85% across all epsilon values and augmentation counts
   - Demonstrates that ICA+FGSM preserves task-relevant EEG features while introducing controlled variations
   - No performance degradation indicates augmentations are physiologically plausible

2. **Uncertainty Quantification (Bottom Panels)**:
   - Mean prediction entropy ranges from 0.72–0.78 across configurations
   - Slight reduction in uncertainty with higher augmentation counts suggests improved confidence calibration
   - Consistent entropy levels enable reliable uncertainty estimation for safety-critical BCI applications

3. **Method Validation**:
   - Flat accuracy curves across augmentation parameters confirm that perturbations do not introduce spurious patterns
   - Stable uncertainty metrics validate the ensemble aggregation approach using cosine similarity

**Scientific Interpretation:**
The consistent performance across parameters indicates that ATCNet already learns robust features from motor imagery EEG. The primary contribution of this method lies in:
- **Uncertainty quantification** for model confidence estimation
- **Physiologically-informed augmentation** framework preserving signal validity
- **Systematic comparison** of augmentation strategies for BCI research

---

#### Technical Contributions:

✅ **Novel Pipeline**: Combined adversarial methods (FGSM) with neuroscience constraints (ICA, frequency filtering)  
✅ **Physiological Validity**: Enforced realistic signal characteristics through bandpass filtering and topographic constraints  
✅ **Comprehensive Evaluation**: Ablation study across epsilon values and augmentation counts with 10-run validation  
✅ **Uncertainty Framework**: Established baseline entropy metrics for 4-class motor imagery tasks

---

## Metrics Used

| Metric           | Description |
|------------------|-------------|
| **Mean Accuracy** | Average correct predictions across subjects |
| **Accuracy Variance** | Measures model performance variability across subjects |
| **Mean Kappa**   | Agreement between predicted and true labels |
| **Kappa Variance** | Reliability difference across subjects |
| **Entropy**      | Uncertainty in prediction probabilities |

---



