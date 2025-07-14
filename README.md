# Spectral Feature Extraction for Robust Network Intrusion Detection Using MFCCs

## Introduction
### Background
The rapid expansion of Internet of Things (IoT) networks has introduced significant security vulnerabilities, as these networks often transmit sensitive data across interconnected devices. Common threats include Denial of Service (DoS), Distributed DoS (DDoS), spoofing, and data theft. Traditional security measures like firewalls and encryption are insufficient against evolving attacks, necessitating advanced Intrusion Detection Systems (IDS). Machine Learning (ML) and Deep Learning (DL) have emerged as powerful tools for IDS, but challenges remain in detecting novel attacks and handling noisy, high-dimensional IoT traffic data

### Problem Statement
Existing IDS face limitations in:

1. Feature extraction: Traditional methods struggle to capture temporal and spectral patterns in IoT traffic.

2. Adaptability: Fixed feature representations (e.g., static MFCCs) may not generalize across diverse attack types.

3. Detection accuracy: Models often fail to distinguish subtle anomalies, especially in multiclass scenarios

### Team Contributions
The paper proposes a novel MFCC-ResNet-18 framework for IoT intrusion detection, with four key contributions:

1. Learnable MFCCs: Adaptive spectral feature extraction via trainable Mel filter banks and DCT, enhancing discriminative power.

2. Cross-domain adaptation: MFCCs, originally for speech, are repurposed to model IoT traffic as acoustic signals.

3. State-of-the-art performance: Achieves 99.9% F1-score on IoTID20 and 100% on NSL-KDD, outperforming transformers and traditional ResNet.

4. Theoretical innovation: Formulates MFCCs as a kernel method, linking feature extraction to kernel-based ML theory.
