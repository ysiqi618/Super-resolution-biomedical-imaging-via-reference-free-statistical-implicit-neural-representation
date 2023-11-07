close all; clc; clear all;
run '~/Desktop/irt/setup.m';
%%
targetSize = 400; % crop/pad img to 400x400

folderPath = '~/Desktop/us_images/';

% Get a list of all JPG files in the folder
jpgFiles = dir(fullfile(folderPath, '*.jpg'));

% Loop through each JPG file
for i = 1:length(jpgFiles)
    % Load the image
    img = imread(fullfile(folderPath, jpgFiles(i).name));
    % Get the dimensions of the loaded image
    [height, width, ~] = size(img);
    img2 = im2double(img);
%    figure; imshow(img2);
    img2 = img2(:,:,1);

    % Calculate cropping or padding dimensions
    padVertical = max(0, floor((targetSize - height) / 2));
    padHorizontal = max(0, floor((targetSize - width) / 2));
    
    cropTop = max(0, floor((height - targetSize) / 2));
    cropLeft = max(0, floor((width - targetSize) / 2));
    
    % Crop or pad the image
    if height >= targetSize && width >= targetSize
        resized_img = img2(cropTop+1:cropTop+targetSize, cropLeft+1:cropLeft+targetSize);
    elseif  height >= targetSize && width < targetSize
        resized_img = img2(cropTop+1:cropTop+targetSize, :);
	pad_img = zeros(targetSize, targetSize);
	pad_img(:, targetSize/2-width/2+1:targetSize/2+width/2) = resized_img;
	resized_img = pad_img;
    elseif  width >= targetSize && height < targetSize
        resized_img = img2(:, cropLeft+1:cropLeft+targetSize);
	pad_img = zeros(targetSize, targetSize);
	pad_img(targetSize/2-height/2+1:targetSize/2+height/2, :) = resized_img;
	resized_img = pad_img;
    else
	img_x = size(img2,1);
        img_y = size(img2,2);
        resized_img = zeros(targetSize,targetSize);
        resized_img(targetSize/2-img_x/2+1:targetSize/2+img_x/2, targetSize/2-img_y/2+1:targetSize/2+img_y/2) = img2;
    end   
    
    figure(1);imshow(resized_img)
    % imwrite(img_pad, './09-53-51-pad.tif','tiff');
    imwrite(resized_img, sprintf('./resized_img_all/img_%d.tif', i),'tiff');
end
