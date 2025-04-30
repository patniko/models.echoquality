# EchoQuality

A tool for assessing the quality of echocardiogram videos using deep learning.

## Overview

This project uses a pre-trained R(2+1)D model to classify the quality of echocardiogram videos. The model analyzes DICOM files and predicts whether the video quality is acceptable.

## Dependencies

- torch
- numpy
- tqdm
- opencv-python
- pydicom
- torchvision

## Usage

The main script `EchoPrime_qc.py` processes DICOM files from a specified directory and outputs quality predictions.

```python
# Example usage
python EchoPrime_qc.py
```

The script will process all DICOM files in the `./model_data/example_study` directory by default.
