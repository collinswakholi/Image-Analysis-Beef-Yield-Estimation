% convert labels from binary/label figure to uint8

clear; close all; clc

Imdir = uigetdir; %get image directory

imds = imageDatastore(Imdir);
cc = imds.Files;

for i = 1:length(cc)
    I = imread(cc{i,1});
    II = 255*(mat2gray(I));
    imwrite(II,cc{i,1})
end
