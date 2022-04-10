%% Wavelet-Domain Image Denoising

clear all;
close all;
clc;

% Load Image
img = imread('test_image.jpg');
img = rgb2gray(img);
img = im2double(img);
[img_height, img_width, img_channel] = size(img);
img_raw = img;
figure();
imshow(img);
title('Original Image (test image.jpg)');
imwrite(img, 'img_ori.jpg');

% Apply Noisy
img_noisy = imnoise(img, 'gaussian', 0, 0.01);
figure();
imshow(img_noisy);
title('Noisy Image');
imwrite(img_noisy, 'img_noisy.jpg');

% Wavelet Filters (Level 2)
L = 2;
[c, s] = wavedec2(img, L, 'bior2.2');
[c_h, c_w] = size(c)
M = length(c)
s
N1 = s(1,1)*s(1,2)
N2 = s(3,1)*s(3,2)

LL = c(1:N1);
LL_image = reshape(LL, s(1,1), s(1,2));
LL_image = LL_image / max(LL);
figure();
imshow(LL_image);
title('LL Image');
imwrite(LL_image, 'LL_image.jpg');

for i= 1:3
    L2H=c(i*N1+1:(i+1)*N1);
    L2H=abs(L2H);
    L2H=L2H/max(L2H);
    L2H_image=reshape(L2H,s(1,1),s(1,2));
    figure();
    imshow(L2H_image);
    title(sprintf('Level 2 LH %d.jpg',i)); 
    imwrite(L2H_image, sprintf('L2H_image_%d.jpg',i));
end

for i= 1:3
    L1H=c(4*N1+(i-1)*N2+1:4*N1+i*N2);
    L1H=abs(L1H);
    L1H=L1H/max(L1H);
    L1H_image=reshape(L1H,s(3,1),s(3,2));
    figure();
    imshow(L1H_image);
    title(sprintf('Level 1 LH %d.jpg',i)); 
    imwrite(L1H_image, sprintf('L1H_image_%d.jpg',i));
end

% Denoising
L = 2;
[c, s] = wavedec2(img_noisy, L, 'bior2.2');
[c_h, c_w] = size(c);
M = length(c);
s;
N1 = s(1,1)*s(1,2);
N2 = s(3,1)*s(3,2);

th = 0.3;
c_hard = zeros(c_h, c_w);
for n=1:N1
    c_hard(n) = c(n);
end
for n=N1+1:M
    if(abs(c(n))<th)
        c_hard(n)=0;
    else
        if(c(n)>0)
            c_hard(n)=c(n)-th;
        end
        if(c(n)<0)
            c_hard(n)=c(n)+th;
        end
    end
end

img_denoised_hard = waverec2(c_hard,s,'bior2.2');
figure();
imshow(img_denoised_hard);
title('denoised image with hard thresholding');
imwrite(img_denoised_hard, 'img_denoised_hard.jpg');

th = 0.4;
c_soft = zeros(c_h, c_w);
for n=1:N1
    c_soft(n) = c(n);
end
for n=N1+1:M
    if(abs(c(n))<th)
        c_soft(n)=0;
    else
        c_soft(n)=c(n);
    end
end
img_denoised_soft = waverec2(c_soft,s,'bior2.2');
figure();
imshow(img_denoised_soft);
title('denoised image with soft thresholding');
imwrite(img_denoised_soft, 'img_denoised_soft.jpg');


%% Evaluation

psnr_noise = psnr(img_noisy, img_raw);
psnr_hard = psnr(img_denoised_hard, img_raw);
psnr_soft = psnr(img_denoised_soft, img_raw);

fprintf('PSNR between orignal image and noisy image : %d dB\n', psnr_noise);
fprintf('PSNR between orignal image and Wavelet filtered image with hard thresholding : %d dB\n', psnr_hard);
fprintf('PSNR between orignal image and Wavelet filtered image with soft thresholding : %d dB\n\n', psnr_soft);

ssim_noise = ssim(img_noisy, img_raw);
ssim_hard = ssim(img_denoised_hard, img_raw);
ssim_soft = ssim(img_denoised_soft, img_raw);

fprintf('SSIM between orignal image and noisy image : %d \n', ssim_noise);
fprintf('SSIM between orignal image and Wavelet filtered image with hard thresholding : %d \n', ssim_hard);
fprintf('SSIM between orignal image and Wavelet filtered image with soft thresholding : %d \n', ssim_soft);

