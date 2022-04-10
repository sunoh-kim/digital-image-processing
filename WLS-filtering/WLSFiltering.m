%% double() 대신에 im2double() 사용하기.. double()하면 이상하게됨.

%% Denoising using Weighted Least Squares Filter

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

img_noisy = imnoise(img,'gaussian');
figure();   
imshow(img_noisy);
title('Noisy Image (test image.jpg)');
imwrite(img_noisy, 'img_noisy.jpg');

eps = 0.1;
lambda = 1;
alpha = 1;
eps2 = 0.1;
img_filtered = wls_filter(img_noisy, eps, lambda, alpha, eps2, img_height, img_width, img_channel);

figure();
imshow(img_filtered);
title('WLS Filtered Image (test image.jpg)');
imwrite(img_filtered, 'img_filtered.jpg');

psnr_noisy = psnr(img_noisy, img_raw);
psnr_WLS = psnr(img_filtered, img_raw);

fprintf('PSNR between orignal image and noisy image : %d dB\n', psnr_noisy);
fprintf('PSNR between orignal image and WLS filtered image : %d dB\n', psnr_WLS);


%% Image Enhancement using Weighted Least Squares Filter

clear all;
close all;
clc;

img = imread('grandcanal.PNG');
img = im2double(img);
[img_height, img_width, img_channel] = size(img);
img_raw = img;
figure();
imshow(img);
title('Original Image (grandcanal.PNG)');

YCBCR = rgb2ycbcr(img_raw);
figure();
imshow(YCBCR);
title('Original Image in YCbCr Color Space (grandcanal.PNG)');
imwrite(YCBCR, 'YCBCR.jpg');
Y = YCBCR(:,:,1);

% WLS filter
eps = 0.1;
lambda = 1;
alpha = 1;
eps2 = 0.1;
eps3 = 0.1;
I = wls_filter(Y, eps, lambda, alpha, eps2, img_height, img_width, 1);
figure();
imshow(I);
title('Estimated Illumination using WLS Filter (grandcanal.PNG)');
imwrite(I, 'Illumination_WLS.jpg');

R = Y./(I+eps3);
figure();
imshow(R);
title('Estimated Reflectance using WLS Filter (grandcanal.PNG)');
imwrite(R, 'Reflectance_WLS.jpg');

img_enhanced = zeros(img_height, img_width, img_channel);
img_enhanced(:,:,1) = img_raw(:,:,1)./(I+eps3);
img_enhanced(:,:,2) = img_raw(:,:,2)./(I+eps3);
img_enhanced(:,:,3) = img_raw(:,:,3)./(I+eps3);
figure();
imshow(img_enhanced);
title('Enhanced Image using WLS Filter (grandcanal.PNG)');
imwrite(img_enhanced, 'img_enhanced.jpg');

% Apply Gaussian Filter

window_size = 19;
sigma_d = 1.5;

I_GF = gaussian_filter(Y, window_size, sigma_d);
figure();
imshow(I_GF);
title('Estimated Illumination using Gaussian Filter (grandcanal.PNG)');
imwrite(I_GF, 'Illumination_GF.jpg');

img_enhanced_GF = zeros(img_height, img_width, img_channel);
img_enhanced_GF(:,:,1) = img_raw(:,:,1)./(I_GF+eps3);
img_enhanced_GF(:,:,2) = img_raw(:,:,2)./(I_GF+eps3);
img_enhanced_GF(:,:,3) = img_raw(:,:,3)./(I_GF+eps3);
figure();
imshow(img_enhanced_GF);
title('Enhanced Image using Gaussian Filter (grandcanal.PNG)');
imwrite(img_enhanced_GF, 'img_enhanced_GF.jpg');

% Apply Bilateral Filter

sigma_d = 1.5;
sigma_s = 0.13;

I_BF(:, :, 1) = bilateral_filter(Y, window_size, sigma_s, sigma_d);
figure();
imshow(I_BF);
title('Estimated Illumination using Bilateral Filter (grandcanal.PNG)');
imwrite(I_BF, 'Illumination_BF.jpg');

img_enhanced_BF = zeros(img_height, img_width, img_channel);
img_enhanced_BF(:,:,1) = img_raw(:,:,1)./(I_BF+eps3);
img_enhanced_BF(:,:,2) = img_raw(:,:,2)./(I_BF+eps3);
img_enhanced_BF(:,:,3) = img_raw(:,:,3)./(I_BF+eps3);
figure();
imshow(img_enhanced_BF);
title('Enhanced Image using Bilateral Filter (grandcanal.PNG)');
imwrite(img_enhanced_BF, 'img_enhanced_BF.jpg');

%% Weighted Least Squares Filter
function img_filtered = wls_filter(img_noisy, eps, lambda, alpha, eps2, img_height, img_width, img_channel)

if img_channel == 3
    img_noisy_r = img_noisy(:, :, 1);
    img_noisy_g = img_noisy(:, :, 2);
    img_noisy_b = img_noisy(:, :, 3);

    img_noisy_r_v = img_noisy_r(:);
    img_noisy_g_v = img_noisy_g(:);
    img_noisy_b_v = img_noisy_b(:);

    vector_length = img_height * img_width;
    L_r = log(img_noisy_r_v + eps);
    L_g = log(img_noisy_g_v + eps);
    L_b = log(img_noisy_b_v + eps);

    % Make Dy
    B = zeros(vector_length, 2);
    diag1 = ones(vector_length, 1);
    diag2 = diag1;
    B(:, 1) = diag1;
    B(:, 2) = -diag2;
    d= [0, -1];
    Dy = spdiags(B, d, vector_length, vector_length);
    [i, j, s] = find(Dy);
    k = size(i);
    for n =1:k
        if(mod(i(n), img_height) == 1)
            s(n)=0;
        end
    end
    Dy = sparse(i,j,s, vector_length, vector_length);

    % Make Dx
    d = [0, img_height];
    Dx = spdiags(B, d, vector_length, vector_length);
    [i, j, s] = find(Dx);
    k = size(i);
    for n =1:k
        if(i(n) > vector_length - img_height)        
            s(n)=0;
        end
    end
    Dx = sparse(i,j,s, vector_length, vector_length);

    % Computer Ax and Ay
    ax_r = lambda./(abs(Dx*L_r).^alpha+eps2);
    ax_g = lambda./(abs(Dx*L_g).^alpha+eps2);
    ax_b = lambda./(abs(Dx*L_b).^alpha+eps2);

    ay_r = lambda./(abs(Dy*L_r).^alpha+eps2);
    ay_g = lambda./(abs(Dy*L_g).^alpha+eps2);
    ay_b = lambda./(abs(Dy*L_b).^alpha+eps2);

    Ax_r = spdiags(ax_r, 0, vector_length, vector_length);
    Ax_g = spdiags(ax_g, 0, vector_length, vector_length);
    Ax_b = spdiags(ax_b, 0, vector_length, vector_length);
    Ay_r = spdiags(ay_r, 0, vector_length, vector_length);
    Ay_g = spdiags(ay_g, 0, vector_length, vector_length);
    Ay_b = spdiags(ay_b, 0, vector_length, vector_length);

    % Solve the linear equation
    D_r = Dx'*Ax_r*Dx+Dy'*Ay_r*Dy;
    D_g = Dx'*Ax_g*Dx+Dy'*Ay_g*Dy;
    D_b = Dx'*Ax_b*Dx+Dy'*Ay_b*Dy;

    D_r = speye(vector_length)+D_r;
    D_g = speye(vector_length)+D_g;
    D_b = speye(vector_length)+D_b;

    yy_r = D_r\img_noisy_r_v;
    yy_g = D_g\img_noisy_g_v;
    yy_b = D_b\img_noisy_b_v;

    img_filtered = zeros(img_height, img_width, img_channel);
    img_filtered(:,:,1) = max(min(reshape(yy_r, img_height, img_width), 1), 0);
    img_filtered(:,:,2) = max(min(reshape(yy_g, img_height, img_width), 1), 0);
    img_filtered(:,:,3) = max(min(reshape(yy_b, img_height, img_width), 1), 0);
end

if img_channel == 1

    img_noisy_v = img_noisy(:);
    vector_length = img_height * img_width;
    L = log(img_noisy_v + eps);

    % Make Dy
    B = zeros(vector_length, 2);
    diag1 = ones(vector_length, 1);
    diag2 = diag1;
    B(:, 1) = diag1;
    B(:, 2) = -diag2;
    d= [0, -1];
    Dy = spdiags(B, d, vector_length, vector_length);
    [i, j, s] = find(Dy);
    k = size(i);
    for n =1:k
        if(mod(i(n), img_height) == 1)
            s(n)=0;
        end
    end
    Dy = sparse(i,j,s, vector_length, vector_length);

    % Make Dx
    d = [0, img_height];
    Dx = spdiags(B, d, vector_length, vector_length);
    [i, j, s] = find(Dx);
    k = size(i);
    for n =1:k
        if(i(n) > vector_length - img_height)        
            s(n)=0;
        end
    end
    Dx = sparse(i,j,s, vector_length, vector_length);

    % Computer Ax and Ay
    ax = lambda./(abs(Dx*L).^alpha+eps2);
    ay = lambda./(abs(Dy*L).^alpha+eps2);

    Ax = spdiags(ax, 0, vector_length, vector_length);
    Ay = spdiags(ay, 0, vector_length, vector_length);

    % Solve the linear equation
    D = Dx'*Ax*Dx+Dy'*Ay*Dy;
    D = speye(vector_length)+D;
    yy = D\img_noisy_v;

    img_filtered = max(min(reshape(yy, img_height, img_width), 1), 0);

end

end


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
