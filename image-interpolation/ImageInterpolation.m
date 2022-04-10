%% double() 대신에 im2double() 사용하기.. double()하면 이상하게됨.

%% Image interpolation by Nyquist filter and comparison with Bicubic interpolation

clear all;
close all;
clc;

% Loading Image
img = imread('test_image.jpg');
img = im2double(img);
[img_height, img_width, img_channel] = size(img);
img_raw = img;
figure();
imshow(img);
title('Original Image (test image.jpg)');

% Apply Gaussian Filter
window_size = 5;
sigma_d = 1.5;

img_gaussian_filtered = zeros(img_height, img_width, img_channel);
img_gaussian_filtered(:,:,1) = gaussian_filter(img(:,:,1), window_size, sigma_d);
img_gaussian_filtered(:,:,2) = gaussian_filter(img(:,:,2), window_size, sigma_d);
img_gaussian_filtered(:,:,3) = gaussian_filter(img(:,:,3), window_size, sigma_d);
figure();
imshow(img_gaussian_filtered);
title('Filtered Image using Gaussian Filter');
imwrite(img_gaussian_filtered, 'img_gaussian_filtered.jpg');

% Decimation
img_decimated = zeros(img_height/4, img_width/4, img_channel);
for i = 1:img_height/4
    for j = 1:img_width/4
        img_decimated(i,j,:) = img_gaussian_filtered(i*4, j*4, :);
    end
end
figure();
imshow(img_decimated);
title('Decimated Image');
imwrite(img_decimated, 'img_decimated.jpg');

% Expansion
img_expanded = zeros(img_height, img_width, img_channel);
for i = 1:img_height/4
    for j = 1:img_width/4
        img_expanded(i*4,j*4,:) = img_decimated(i, j, :);
    end
end
figure();
imshow(img_expanded);
title('Expanded Image');
imwrite(img_expanded, 'img_expanded.jpg');

% Nyquist Filter
Nyquist_filter_file = load('Nyquist_filter.mat');
Nyquist_filter_coefficients = 4*Nyquist_filter_file.Nyquist_filter;
Nyquist_filter = Nyquist_filter_coefficients' * Nyquist_filter_coefficients;
img_nyquist = imfilter(img_expanded, Nyquist_filter, 'symmetric', 'same');
figure();
imshow(img_nyquist);
title('Image after Interpolation by Nyquist Filter');
imwrite(img_nyquist, 'img_nyquist.jpg');

% Bicubic Interpolation
img_bicubic = imresize(img_decimated, 4);
figure();
imshow(img_bicubic);
title('Image after Bicubic Interpolation');
imwrite(img_bicubic, 'img_bicubic.jpg');

%% Evaluation

psnr_Nyquist = psnr(img_nyquist, img_raw);
psnr_Bicubic = psnr(img_bicubic, img_raw);

fprintf('PSNR between orignal image and Nyquist filtered image : %d dB\n', psnr_Nyquist);
fprintf('PSNR between orignal image and Bicubic interpolated image : %d dB\n\n', psnr_Bicubic);

ssim_Nyquist = ssim(img_nyquist, img_raw);
ssim_Bicubic = ssim(img_bicubic, img_raw);

fprintf('SSIM between orignal image and Nyquist filtered image : %d \n', ssim_Nyquist);
fprintf('SSIM between orignal image and Bicubic interpolated image : %d \n', ssim_Bicubic);

%% Gaussian Filter 
function output = gaussian_filter(img, window_size, sigma_d)

[img_height, img_width] = size(img);
offset = (window_size - 1) / 2;
img_padded = padarray(img, [offset, offset], 'both', 'replicate');

output = zeros([img_height, img_width]);

[x, y] = meshgrid(-offset:offset, -offset:offset);
distance_kernel = exp(-(x.^2 + y.^2) / (2 * sigma_d^2));

for i = 1+offset:img_height+offset
    for j = 1+offset:img_width+offset
        row_min = i - offset;
        row_max = i + offset;
        column_min = j - offset;
        column_max = j + offset;
        sub_img = img_padded(row_min:row_max, column_min:column_max);
        
        norm = sum(distance_kernel(:));
        output(i - offset, j - offset) = sum(sum(distance_kernel.*sub_img)) / norm;
    end
end

end
