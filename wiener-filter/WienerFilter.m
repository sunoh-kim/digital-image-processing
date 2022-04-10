clear all;
close all;
clc;

%% Loading Image & Making Blur and Noise

% Loading Image
img = imread('test_image.jpg');
img = rgb2gray(img);
img = im2double(img);
[img_height, img_width] = size(img);
imwrite(img, 'img_original.jpg');
figure()
imshow(img);
title('Original Image');

% Making Blur
PSF = fspecial('motion', 15, 135);
[filter_size, ~] = size(PSF);
size_pad = (filter_size - 1) / 2;
img_padded = padarray(img, [size_pad, size_pad], 'both', 'symmetric');
img_blurred = conv2(img_padded, PSF);
img_blurred = img_blurred(1+size_pad*2:img_height+size_pad*2, 1+size_pad*2:img_width+size_pad*2);
imwrite(img_blurred, 'img_blurred.jpg');
figure()
imshow(img_blurred)
title('Blurred Image')

% % Making Noise
% noise_intensity = 0.1;
% img_noisy = img + noise_intensity * randn(img_height, img_width);
% img_noisy = min(1, img_noisy);
% img_noisy = max(0, img_noisy);
% imwrite(img_noisy, 'img_noisy.jpg');
% figure()
% imshow(img_noisy)
% title('Noisy Image');

%% Deblurring by Wiener Filter

eps = 0.05;

% Preprocessing
[filter_size, ~] = size(PSF);
size_pad = (filter_size - 1) / 2;
size_zero_pad = 100;
% img_blurred_padded = padarray(img_blurred, [size_pad, size_pad], 'both', 'symmetric');
img_blurred_padded = padarray(img_blurred, [size_zero_pad, size_zero_pad], 0, 'both');
[img_padded_height, img_padded_width] = size(img_blurred_padded);
imwrite(img_blurred_padded, 'img_blurred_with_zero_padding.jpg');
figure()
imshow(img_blurred_padded)
title('Blurred Image with Zero Padding')

af = padarray(img_blurred_padded, [filter_size-1, filter_size-1], 0, 'post');
hf = padarray(PSF, [img_padded_height-1, img_padded_width-1], 0, 'post');

% Deblurring
YF = fft2(af);
HF = fft2(hf);
D = HF.*conj(HF) + eps;
W = conj(HF)./D;
XH = W.*YF;
xh = ifft2(XH);
% img_deblurred_wiener = real(xh(1+size_pad+size_zero_pad:img_height+size_pad+size_zero_pad, ...
%     1+size_pad+size_zero_pad:img_width+size_pad+size_zero_pad));
img_deblurred_wiener = real(xh(1+size_zero_pad:img_height+size_zero_pad, ...
    1+size_zero_pad:img_width+size_zero_pad));

imwrite(img_deblurred_wiener, 'img_deblurred_wiener.jpg');
figure()
imshow(img_deblurred_wiener)
title('Image Deblurred by Wiener Filter')

%% Deblurring by Matlab Functions

eps = 0.05;
% Preprocessing
img_blurred_tapered = edgetaper(img_blurred, PSF);

% Deblurring by built-in Wiener
img_deblurred_matlab_wiener = deconvwnr(img_blurred_tapered, PSF, eps);
imwrite(img_deblurred_matlab_wiener, 'img_deblurred_matlab_wiener.jpg');
figure()
imshow(img_deblurred_matlab_wiener)
title('Image Deblurred by Matlab Wiener Filter');

% Deblurring by built-in Lucy-Richardson
img_deblurred_matlab_lucy = deconvlucy(img_blurred_tapered, PSF);
imwrite(img_deblurred_matlab_lucy, 'img_deblurred_matlab_lucy.jpg');
figure()
imshow(img_deblurred_matlab_lucy)
title('Image Deblurred by Matlab Lucy-Richardson Filter');

%% Denoising

noise_mean = 0;
noise_var = 0.01;
img_noisy = imnoise(img,'gaussian',noise_mean,noise_var);
imwrite(img_noisy, 'img_noisy.jpg');
figure()
imshow(img_noisy)
title('Noisy Image');

img_denoised = wiener2(img_noisy, [3 3]);
imwrite(img_denoised, 'img_denoised.jpg');
figure()
imshow(img_denoised)
title('Image Denoised by Weiner Filter')

psnr_noisy = psnr(img_noisy, img);
psnr_denoised = psnr(img_denoised, img);

fprintf('PSNR between orignal image and noisy image : %d dB\n', psnr_noisy);
fprintf('PSNR between orignal image and denoised image : %d dB\n', psnr_denoised);

