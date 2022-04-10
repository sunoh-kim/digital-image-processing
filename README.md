# Digital Image Processing - Assignment Solutions


This repository contains my assignment solutions for the Digital Image Processing course (M2608.001000_001) offered by Seoul National University (Fall 2020).

The algorithms for the assignments are implemented using MATLAB.


### Demosaicing:
Demosaicing is the process of reconstructing a full color image from the incomplete color samples output from an image sensor overlaid with a color filter array (CFA).

### Edge Enhancement:
Edge enhancement is an image processing technique that enhances the edge contrast of an image or video in an attempt to improve its acutance (apparent sharpness). I used box filters and a Gaussian filter for edge enhancement.

### Bilateral Filter:
A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. I used it for denoising.

### Wiener Filter:
A Wiener filter is a filter used to produce an estimate of a desired or target random process by linear time-invariant (LTI) filtering of an observed noisy process, assuming known stationary signal and noise spectra, and additive noise. I used it for denoising and deblurring.

### Nonlinear Processing:
Nonlinear processing produces output that is not a linear function of its input. Here, I used histogram equalization for contrast adjustment and median filter for denoising.

### WLS Filter:
A Weighted Least Squares (WLS) filter is a well-known edge preserving smoothing technique. I used it for image enhancement and denoising.

### Image Interpolation:
For image interpolation, I used Nyquist Filter Design in MATLAB.

### Wavelet Filter:
A Wavelet filter decomposes the signal using the DWT, filter the signal in the wavelet space using thresholding, and invert the filtered signal to reconstruct the original signal. I used it for denoising.

### JPEG Compression:
For JPEG Compression, I used Discrete Cosine Transformation (DCT). The DCT expresses a finite sequence of data points in terms of a sum of cosine functions oscillating at different frequencies.
