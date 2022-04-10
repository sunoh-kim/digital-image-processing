clear all;
close all;
clc;

%% Load Image & Make Noise

img = imread('test_image.png');
img = im2double(img);
[img_height, img_width, channel] = size(img);
figure()
imshow(img)
title('Original Image');

var = 0.1;
img_noised = img + var * randn(img_height, img_width, channel);
img_noised = min(1, img_noised);
img_noised = max(0, img_noised);
figure()
imshow(img_noised)
imwrite(img_noised, 'img_noised.jpg');
title('Noised Image');

img_noised_r = img_noised(:,:,1);
img_noised_g = img_noised(:,:,2);
img_noised_b = img_noised(:,:,3);

%% Apply Gaussian Filter

window_size = 19;
sigma_d = 4;
img_Gaussian_filtered = zeros([img_height, img_width, channel]);

img_Gaussian_filtered(:, :, 1) = gaussian_filter(img_noised_r, window_size, sigma_d);
img_Gaussian_filtered(:, :, 2) = gaussian_filter(img_noised_g, window_size, sigma_d);
img_Gaussian_filtered(:, :, 3) = gaussian_filter(img_noised_b, window_size, sigma_d);

figure();
imshow(img_Gaussian_filtered);
title('Gaussian Filtered Image');
imwrite(img_Gaussian_filtered, 'img_Gaussian_filtered.jpg');

%% Apply Bilateral Filter

sigma_d = 4;
sigma_s = 0.5;
img_Bilateral_filtered = zeros([img_height, img_width, channel]);

img_Bilateral_filtered(:, :, 1) = bilateral_filter(img_noised_r, window_size, sigma_s, sigma_d);
img_Bilateral_filtered(:, :, 2) = bilateral_filter(img_noised_g, window_size, sigma_s, sigma_d);
img_Bilateral_filtered(:, :, 3) = bilateral_filter(img_noised_b, window_size, sigma_s, sigma_d);

figure();
imshow(img_Bilateral_filtered);
title('Bilateral Filtered Image');
imwrite(img_Bilateral_filtered, 'img_Bilateral_filtered.jpg');

%% Evaluation

psnr_noise = psnr(img_noised, img);
psnr_Gaussian_filter = psnr(img_Gaussian_filtered, img);
psnr_Bilateral_filter = psnr(img_Bilateral_filtered, img);

fprintf('PSNR between orignal image and noised image : %d dB\n', psnr_noise);
fprintf('PSNR between orignal image and Gaussian filtered image : %d dB\n', psnr_Gaussian_filter);
fprintf('PSNR between orignal image and Bilateral filtered image : %d dB\n\n', psnr_Bilateral_filter);

ssim_noise = ssim(img_noised, img);
ssim_Gaussian_filter = ssim(img_Gaussian_filtered, img);
ssim_Bilateral_filter = ssim(img_Bilateral_filtered, img);

fprintf('SSIM between orignal image and noised image : %d \n', ssim_noise);
fprintf('SSIM between orignal image and Gaussian filtered image : %d \n', ssim_Gaussian_filter);
fprintf('SSIM between orignal image and Bilateral filtered image : %d \n', ssim_Bilateral_filter);


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

%% Bilateral Filter 
function output = bilateral_filter(img, window_size, sigma_s, sigma_d)

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
        
        value_kernel = exp(-(sub_img - img_padded(i, j)).^2 / (2 * sigma_s^2));
        bilateral_kernel = value_kernel .* distance_kernel;
        
        norm = sum(bilateral_kernel(:));
        output(i - offset, j - offset) = sum(sum(bilateral_kernel.*sub_img)) / norm;
    end
end

end

