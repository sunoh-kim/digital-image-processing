%% double() 대신에 im2double() 사용하기.. double()하면 이상하게됨.

%% Histogram Equalization (head.gif)

clear all;
close all;
clc;

% Loading Image
img = imread('head.gif');
img = im2double(img);
[img_height, img_width] = size(img);
figure();
imshow(uint8(img));
title('Original Image (head.gif)');

L = 0:255; 
hist = zeros(1,256);
for i = 1:img_height
    for j = 1:img_width
        for k = 1:256
            if img(i,j) == L(k)
                hist(k) = hist(k) + 1;
            end
        end
    end
end
% hist(:, 1) = 0;
figure();
plot(hist);
title('Histogram (head.gif)');

c_prob = hist/(img_height*img_width);
cc_prob = cumsum(c_prob);
figure();
plot(cc_prob);
title('CDF of The Histogram (head.gif)');

% converting these cumulative intensities into integers 
out_int = zeros(1,256);
for i = 1:256
    out_int(i) = floor(((cc_prob(i) - min(cc_prob))/(1 - min(cc_prob)))*255 + 0.5);
end

img_out = zeros(img_height,img_width);
for i = 1:img_height
    for j = 1:img_width
        img_out(i,j) = out_int(img(i,j) + 1);
    end
end
figure();
imshow(uint8(img_out));
title('Histogram Equalized image (head.gif)');

L = 0:255; 
hist2 = zeros(1,256);
for i = 1:img_height
    for j = 1:img_width
        for k = 1:256
            if img_out(i,j) == L(k)
                hist2(k) = hist2(k) + 1;
            end
        end
    end
end
figure();
plot(hist2);
title('Histogram after Histogram Equalization (head.gif)');

c_prob2 = hist2/(img_height*img_width);
cc_prob2 = cumsum(c_prob2);
figure();
plot(cc_prob2);
title('CDF of The Histogram after Histogram Equalization (head.gif)');


%% Histogram Equalization (low.png)

clear all;
close all;
clc;

% Loading Image
ori_img = imread('low.png');
img = im2double(ori_img);
[img_height, img_width, channel] = size(img);
figure();
imshow(ori_img);
title('Original Image (low.png)');

YCBCR = rgb2ycbcr(ori_img);
figure();
imshow(YCBCR);
title('Original Image in YCbCr Color Space (low.png)');
Y = YCBCR(:,:,1);

L = 0:255; 
hist = zeros(1,256);
for i = 1:img_height
    for j = 1:img_width
        for k = 1:256
            if Y(i,j) == L(k)
                hist(k) = hist(k) + 1;
            end
        end
    end
end
figure();
plot(hist);
title('Histogram (low.png)');

c_prob = hist/(img_height*img_width);
cc_prob = cumsum(c_prob);
figure();
plot(cc_prob);
title('CDF of The Histogram (low.png)');

% converting these cumulative intensities into integers 
out_int = zeros(1,256);
for i = 1:256
    out_int(i) = floor(((cc_prob(i) - min(cc_prob))/(1 - min(cc_prob)))*255 + 0.5);
end

Y_out = zeros(img_height,img_width);
for i = 1:img_height
    for j = 1:img_width
        Y_out(i,j) = out_int(Y(i,j) + 1);
    end
end
YCBCR(:,:,1) = Y_out;
img_out = ycbcr2rgb(YCBCR);

% img_out = zeros(img_height,img_width, channels);
% for i = 1:img_height
%     for j = 1:img_width
%         for k = 1:channels
%             img_out(i,j,k) = out_int(img(i,j,k) + 1);
%         end
%     end
% end

figure();
imshow(uint8(img_out));
title('Histogram Equalized image (low.png)');

%% Median Filter

clear all;
close all;
clc;

% Loading Image
ori_img = imread('test_image.jpg');
figure();
imshow(ori_img);
title('Original Image');

img_noisy = imnoise(ori_img, 'salt & pepper',0.05);
figure();
imshow(img_noisy);
title('Noisy Image');

img = im2double(ori_img);
img_noisy = im2double(img_noisy);
[img_height, img_width, channels] = size(img);

% Bilateral filter
% sigmar = 40;
% eps = 1e-3;
% sigmas = 3;

% sigma_d = 1.5;
% sigma_s = 0.13;
sigma_d = 3;
sigma_s = 0.5;
window_size = 19;
img_Bilateral_filtered = zeros([img_height, img_width, channels]);
img_noised_r = img_noisy(:,:,1);
img_noised_g = img_noisy(:,:,2);
img_noised_b = img_noisy(:,:,3);
img_Bilateral_filtered(:, :, 1) = bilateral_filter(img_noised_r, window_size, sigma_s, sigma_d);
img_Bilateral_filtered(:, :, 2) = bilateral_filter(img_noised_g, window_size, sigma_s, sigma_d);
img_Bilateral_filtered(:, :, 3) = bilateral_filter(img_noised_b, window_size, sigma_s, sigma_d);
yb = img_Bilateral_filtered;
figure();
imshow(yb);
title('Bilateral Filtered Image');

% Median filter
ym = zeros([img_height, img_width, channels]);
ym(:, :, 1) = medfilt2(img_noisy(:,:,1), [5 5]);
ym(:, :, 2) = medfilt2(img_noisy(:,:,2), [5 5]);
ym(:, :, 3) = medfilt2(img_noisy(:,:,3), [5 5]);

figure();
imshow(ym);
title('Median Filtered Image');


psnr_noisy = psnr(img_noisy, img);
psnr_bilateral = psnr(yb, img);
psnr_median = psnr(ym, img);

fprintf('PSNR between orignal image and noisy image : %d dB\n', psnr_noisy);
fprintf('PSNR between orignal image and bilateral filtered image : %d dB\n', psnr_bilateral);
fprintf('PSNR between orignal image and median filtered image : %d dB\n', psnr_median);


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
