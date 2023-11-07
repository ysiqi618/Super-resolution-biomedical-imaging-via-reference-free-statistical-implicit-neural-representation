close all; clc; clear all;
run '~/Desktop/irt/setup.m';
%%
nx = 400; % crop/pad img to 400x400
% img = imread('09-53-51.jpg');
img = imread('09-55-50.jpg');

img2 = im2double(img);
% figure; imshow(img(:,:,1));
% figure; imshow(img(:,:,2));
% figure; imshow(img(:,:,3));
figure; imshow(img2);
img2 = img2(:,:,1);
img_x = size(img2,1);
img_y = size(img2,2);
img_pad = zeros(nx,nx);
img_pad(nx/2-img_x/2+1:nx/2+img_x/2, nx/2-img_y/2+1:nx/2+img_y/2) = img2;
figure;imshow(img_pad)
% imwrite(img_pad, './09-53-51-pad.tif','tiff');
imwrite(img_pad, './09-55-50.tif','tiff');