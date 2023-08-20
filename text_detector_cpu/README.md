# text_detector_cpu

## Introduction

This project compiled inference post-processing code for a pre-trained weight that provide by Openvino.

Below is the list of model that use in this project and its high level description on how post-processing is done.

## Detection

1. Spoof detection using MN3 model based on Celeb Anti-Spoofing dataset pre-trained by some guy using Pytorch converted to ONNX.
  - Need to convert ONNX model to Openvino model by using omz_converter.
  - Apply median filter to reduce image dimension. (This step is not necessary but work great for high resolution image)
  - Adjust the input shape by tranpose the input shape to correct shape for inference.

2. Face detection using face-detection-adas-0001 model pre-trained by Openvino.
  - Same as anti spoof except the model is already in Openvino format.
  - The output is softmax score of face/no face sum to 1.
  - The model isn't good but usable.

3. Text Detection using text-detection-0004 model pre-trained by Openvino based on PixelLink.
  - The weight is in Openvino format.
  - Need to post process to tensor output to get the bounding box.
  - The output from network is 2 tensor, one is the score map of instance segmentation that label text/no text and another is the link localization map that use to connect the text box using 8 direction.
  - Code reference: https://github.com/mayank-git-hub/Text-Recognition/blob/master/src/helper/utils.py. This repo show the implementation of connected component after inference output.
