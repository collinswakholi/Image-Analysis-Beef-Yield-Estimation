clear; clc; close all;

% load image datastores
imds = imageDatastore('out');
mask_folder = 'mask_n';

files = imds.Files;
len = length(files);

for i = 1:len
    I = imread(files{i,1});
    str1 = strsplit(files{i,1},'\');
    name = str1{1,end};
    
    mask = imread([mask_folder,'\',name]);
    mask = perfect_mask(mask, I);
    imwrite(mask,[mask_folder,'\',name])
    i
end
