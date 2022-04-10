%% Load images

img = imread('test_image.jpg');
img = im2double(img);

%% Box Filter

size_BF_1 = 3;
size_BF_2 = 5;

box_filter_1 = ones(size_BF_1, size_BF_1) / size_BF_1^2;
box_filter_2 = ones(size_BF_2, size_BF_2) / size_BF_2^2;

figure();
surf(-1:1, -1:1, box_filter_1)
[freq_response_1, w1_1, w2_1] = freqz2(box_filter_1);
figure();
surf(w1_1, w2_1, abs(freq_response_1))

figure();
surf(-2:2, -2:2, box_filter_2)
[freq_response_2, w1_2, w2_2] = freqz2(box_filter_2);
figure();
surf(w1_2, w2_2, abs(freq_response_2))

%% Gaussian Filter

size_GF = 7;
bandwidth = 3;

gaussian_filter = zeros(size_GF, size_GF);
mean = (size_GF+1) / 2;
for m = 1:size_GF
    for n = 1:size_GF
        gaussian_filter(m, n) = exp((-(m-mean)^2-(n-mean)^2)/(2 * bandwidth^2));
    end
end
s = sum(gaussian_filter(:));
gaussian_filter = gaussian_filter / s;

figure();
surf(-3:3, -3:3, gaussian_filter)
[freq_response_3, w1_3, w2_3] = freqz2(gaussian_filter);
figure();
surf(w1_3, w2_3, abs(freq_response_3))

%% Filter-based Image Enhancement

alpha = 1;
img_enhanced_BF_1 = image_enhancement(img, 'box_filter_1', box_filter_1, size_BF_1, alpha);
img_enhanced_BF_2 = image_enhancement(img, 'box_filter_2', box_filter_2, size_BF_2, alpha);
img_enhanced_GF = image_enhancement(img, 'gaussian_filter', gaussian_filter, size_GF, alpha);

%% Image Enhancement Function
function img_enhanced = image_enhancement(img, name, filter, size_filter, alpha)

[image_height, image_width, channel] = size(img);
size_pad = (size_filter - 1) / 2;
img_padded = padarray(img, [size_pad, size_pad], 'both', 'replicate');

low_pass_signal = zeros([image_height, image_width, channel]);

low_pass_signal_r = conv2(img_padded(:,:,1), filter);
low_pass_signal_g = conv2(img_padded(:,:,2), filter);
low_pass_signal_b = conv2(img_padded(:,:,3), filter);

low_pass_signal(:,:,1) = low_pass_signal_r(1+size_pad:image_height+size_pad, 1+size_pad:image_width+size_pad);
low_pass_signal(:,:,2) = low_pass_signal_g(1+size_pad:image_height+size_pad, 1+size_pad:image_width+size_pad);
low_pass_signal(:,:,3) = low_pass_signal_b(1+size_pad:image_height+size_pad, 1+size_pad:image_width+size_pad);
imwrite(low_pass_signal, sprintf('low_pass_signal_%s.jpg', name));

edge = img - low_pass_signal;
imwrite(edge, sprintf('edge_%s.jpg', name));

img_enhanced = img + alpha * edge;
imwrite(img_enhanced, sprintf('img_enhanced_%s.jpg', name));


end

