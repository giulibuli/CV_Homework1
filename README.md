# Homework 1: Image Enhancement - Computer Vision

## Pipeline Overview
This repository contains a classical image processing pipeline built in Python to enhance images affected by low contrast and non-uniform illumination.

## Requirements
To run this project, you need Python (e.g., Python 3.8+) and the following libraries:
* `opencv-python` (OpenCV)
* `numpy`
* `scikit-image`

You can install the dependencies using:
`pip install opencv-python numpy scikit-image`

## File Structure
* `main.py`: The main script containing the implementation of the degradations and the four enhancement methods.
* `img1.jpg` to `img5.jpg`: Original source images used for the evaluation (Unsplash License).
* `sources.txt`: Links and attribution for the source images.
* `/results/`: Directory generated at runtime containing the degraded images, the outputs of the processed methods, and the `metrics.csv` file.

## How to Run
To reproduce the experiment and generate the images and metrics, run the following command in the terminal:

python main.py

Note: The script uses a fixed random seed to ensure the Gaussian noise degradation is fully reproducible.