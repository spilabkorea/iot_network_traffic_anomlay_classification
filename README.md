# Spectral Feature Extraction for Robust Network Intrusion Detection Using MFCCs

## Introduction
### Background
The rapid expansion of Internet of Things (IoT) networks has introduced significant security vulnerabilities, as these networks often transmit sensitive data across interconnected devices. Common threats include Denial of Service (DoS), Distributed DoS (DDoS), spoofing, and data theft. Traditional security measures like firewalls and encryption are insufficient against evolving attacks, necessitating advanced Intrusion Detection Systems (IDS). Machine Learning (ML) and Deep Learning (DL) have emerged as powerful tools for IDS, but challenges remain in detecting novel attacks and handling noisy, high-dimensional IoT traffic data

### Problem Statement
Existing IDS face limitations in:

1. Feature extraction: Traditional methods struggle to capture temporal and spectral patterns in IoT traffic.

2. Adaptability: Fixed feature representations (e.g., static MFCCs) may not generalize across diverse attack types.

3. Detection accuracy: Models often fail to distinguish subtle anomalies, especially in multiclass scenarios

### Contributions
The paper proposes a novel MFCC-ResNet-18 framework for IoT intrusion detection, with four key contributions:

1. Learnable MFCCs: Adaptive spectral feature extraction via trainable Mel filter banks and DCT, enhancing discriminative power.

2. Cross-domain adaptation: MFCCs, originally for speech, are repurposed to model IoT traffic as acoustic signals.

3. State-of-the-art performance: Achieves 99.9% F1-score on IoTID20 and 100% on NSL-KDD, outperforming transformers and traditional ResNet.

4. Theoretical innovation: Formulates MFCCs as a kernel method, linking feature extraction to kernel-based ML theory.

![](Figures/arch.png?raw=true)

## Requirements
- **Python 3.10.0**
### Python Packages
- **Pytorch=2.7.1**
- **numpy=2.2.6**
- **pandas=2.3.1**
- **scikit-learn==1.7.0**
- **tqdm=4.67.1**
### Benchmark Datasets 📝
The study evaluates the proposed model on three widely used IoT network intrusion detection datasets:

1. [**IoTID20**](https://www.kaggle.com/datasets/rohulaminlabid/iotid20-dataset)

- Source: Collected from home IoT devices (e.g., SKT NGU and EZVIZ Wi-Fi cameras).

- Size: 625,783 records with 86 features.

- Attack Types: Includes various IoT-specific attacks.

2. [**CICIoT2023**](https://www.kaggle.com/datasets/akashdogra/cic-iot-2023)

- Source: Simulated using 105 IoT devices.

- Attack Types: 33 attacks grouped into 7 categories (DDoS, DoS, Mirai, Brute Force, Web-based, Spoofing, Recon)

- Complexity: Designed for large-scale, realistic IoT attack scenarios.

3. [**NSL-KDD**](https://www.kaggle.com/datasets/hassan06/nslkdd)

- Source: Improved version of KDD CUP 1999, addressing redundancy and noise issues.

- Attack Types: 5 categories (Normal, DoS, Probe, U2R, R2L).

- **Usage: Benchmark for ML/DL-based IDS evaluation.**
### Preprocessing 
The following steps were applied to all datasets for consistency and model compatibility:

1. **Label Standardization**

- Converted labels into binary classes: Normal vs. Attack.

- Applied label encoding for numerical representation.
  
2. **Data Cleaning**

- Removed columns with: Infinite or NaN values. Over 50% zero values (sparse features).

- Added a time index for temporal analysis.
  
3. **Normalization**

- Scaled features using MinMaxScaler to [0, 1] range.

4. **Feature Extraction**

- Transformed raw signals into MFCCs (Mel-frequency cepstral coefficients) to capture spectral-temporal patterns.

- Applied PCA for dimensionality reduction post-MFCC extraction.

5. **Data Augmentation**

- Used sampling techniques (e.g., oversampling/undersampling) to address class imbalance.
-The preprocessed data is aviable in code for the training and evaulation purpose.

## Installation

To run the code, follow these steps:

Clone the repository 

```
git@github.com:spilabkorea/iot_network_traffic_anomlay_classification.git
```
Install the libraries by the following command
```
conda env create -f environment.yml
```
### Model Training
In this project, we have employed a ResNet18 with channel and spatial attention. The code is provided in the ipynb file to reproduced the results. 

## Citation
```bibtex
@article{lee2025spectral,
  author       = {HyeYoung Lee},
  title        = {Spectral Feature Extraction for Robust Network Intrusion Detection Using MFCCs},
  journal      = {arXiv preprint},
  year         = {2025},
  url          = {}
}
