%% double() 대신에 im2double() 사용하기.. double()하면 이상하게됨.

clear all;
close all;
clc;

% Load Image
img = imread('test_image.png');
img = rgb2gray(img);
img = imresize(img,[440 704]);
% YCbCr = rgb2ycbcr(img);
% img = YCbCr(:,:, 1);
img = im2double(img);
[img_height, img_width] = size(img);
img_raw = img;
figure();
imshow(img);
title('Original Image (test image.jpg)');
imwrite(img, 'img_ori.jpg');

img_jpg_matlab = imread('img_ori.jpg');
img_jpg_matlab = im2double(img_jpg_matlab);
figure();
imshow(img_jpg_matlab);
title('MATLAB JPEG Image');
imwrite(img_jpg_matlab, 'img_jpg_matlab.jpg');

% DCT
img_dct = dct2(img);
figure();
imshow(img_dct);
title('Energy Compaction by DCT');
imwrite(img_dct, 'img_dct.jpg');

% Quantization Matrix Design
QL =     [16 11 10 16 24 40 51 61; 
            12 12 14 19 26 58 60 55;
            14 13 16 24 40 57 69 56; 
            14 17 22 29 51 87 80 62;
            18 22 37 56 68 109 103 77;
            24 35 55 64 81 104 113 92;
            49 64 78 87 103 121 120 101;
            72 92 95 98 112 100 103 99];

QF = 100;
if QF < 50
    S = 5000/QF;
else
    S = 200-2*QF;
end

Q = S*QL+50;
Q = floor(Q/100);
for n = 1:8
    for m = 1:8
        Q(m,n) = max(Q(m,n),1);
    end
end


% Compression
r = img_height/8;
c = img_width/8;

s = 1;
img_compressed = zeros(img_height, img_width);
for i=1:r
    e = 1;
    for j=1:c
        block = img(s:s+7,e:e+7);
        block_dct = 255*dct2(block);
        block_compressed = zeros(8, 8);
        for x=1:8
            for y=1:8
                block_compressed(x, y) = round(block_dct(x, y)./Q(x, y));
            end
        end
      
        img_compressed(s:s+7,e:e+7) = block_compressed;
        e = e + 8;
     end
     s = s + 8;
end

% Decompression
img_decompressed = zeros(img_height, img_width);
s = 1;
for i=1:r
    e = 1;
    for j=1:c
        block_compressed = img_compressed(s:s+7,e:e+7);
        block_dct = zeros(8, 8);
        for x=1:8
            for y=1:8
                block_dct(x, y) = round(Q(x, y).*block_compressed(x, y))/255; 
            end
        end
        block_idct = idct2(block_dct);

        img_decompressed(s:s+7,e:e+7) = block_idct;
        e = e + 8;
      end
      s = s + 8;
  end

figure();
imshow(img_decompressed);
title('Image Decompressed');
imwrite(img_decompressed, 'img_decompressed.jpg');



%% Evaluation

psnr_jpg_matlab = psnr(img_jpg_matlab, img_raw);
psnr_decompressed = psnr(img_decompressed, img_raw);

fprintf('PSNR between orignal image and matlab JPEG image : %d dB\n', psnr_jpg_matlab);
fprintf('PSNR between orignal image and decompressed image with quality factor %d (mine) : %d dB\n', QF, psnr_decompressed);


