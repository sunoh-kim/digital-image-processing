%% Load images
info = imfinfo('test.dng');
gt_rgb = imread('test_groundtruth.jpg');
gt_rgb = im2double(gt_rgb);

src_cfa = read(Tiff('test.dng'));
src_cfa = double(src_cfa);
pmax = double(max(max(src_cfa)));
fprintf('Maximum pixel value : %d \n', pmax);

test_cfa = src_cfa / pmax;
imwrite(test_cfa, 'test_cfa_before_gamma_encoding.bmp');
test_cfa = sqrt(test_cfa);
imwrite(test_cfa, 'test_cfa_after_gamma_encoding.bmp');

%% Mosaic image
[image_height, image_width, num_chanels] = size(gt_rgb);

syn_cfa = gt_rgb(:,:,2);
syn_cfa(1:2:image_height, 2:2:image_width) = gt_rgb(1:2:image_height, 2:2:image_width,1);
syn_cfa(2:2:image_height, 1:2:image_width) = gt_rgb(2:2:image_height, 1:2:image_width,3);

imwrite(syn_cfa, 'synthetic_cfa.bmp');

%% Demosaic image
demosaic_rgb_from_syn_cfa = demosaicing_Image(syn_cfa);
demosaic_rgb_from_raw_cfa = demosaicing_Image(test_cfa);

assert(all(size(gt_rgb) == size(demosaic_rgb_from_syn_cfa)));
assert(all(size(gt_rgb) == size(demosaic_rgb_from_raw_cfa)));

imwrite(demosaic_rgb_from_syn_cfa, 'demosaic_rgb_from_synthetic_cfa.bmp');
imwrite(demosaic_rgb_from_raw_cfa, 'demosaic_rgb_from_raw_cfa.bmp');

%% Compute PSNR
psnr_from_syn_cfa = psnr(demosaic_rgb_from_syn_cfa, gt_rgb);
psnr_from_raw_cfa = psnr(demosaic_rgb_from_raw_cfa, gt_rgb);

fprintf('PSNR of demosaiced RGB from synthetic CFA : %d dB\n', psnr_from_syn_cfa);
fprintf('PSNR of demosaiced RGB from raw CFA : %d dB\n', psnr_from_raw_cfa);

%% Demosaicing function
function demosaic_image = demosaicing_Image(image)

[image_height, image_width] = size(image);

red_mask = repmat([0 1; 0 0], image_height/2, image_width/2);
green_mask = repmat([1 0; 0 1], image_height/2, image_width/2);
blue_mask = repmat([0 0; 1 0], image_height/2, image_width/2);

image_r = image .* red_mask;
image_g = image .* green_mask;
image_b = image .* blue_mask;

image_r_padded =  padarray(image_r,[2, 2],'replicate', 'both');
image_g_padded =  padarray(image_g,[2, 2],'replicate', 'both');
image_b_padded =  padarray(image_b,[2, 2],'replicate', 'both');

kernel_g = [0, 0.25, 0; 0.25, 1., 0.25; 0, 0.25, 0];
kernel_br = [0.25, 0.5, 0.25; 0.5, 1., 0.5; 0.25, 0.5, 0.25];

demosaic_image = zeros([image_height, image_width, 3]);

demosaic_r = conv2(image_r_padded, kernel_br);
demosaic_g = conv2(image_g_padded, kernel_g);
demosaic_b = conv2(image_b_padded, kernel_br);

demosaic_image(:,:,1) = demosaic_r(3:image_height+2, 3:image_width+2);
demosaic_image(:,:,2) = demosaic_g(3:image_height+2, 3:image_width+2);
demosaic_image(:,:,3) = demosaic_b(3:image_height+2, 3:image_width+2);

end
