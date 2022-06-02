% Rename images in specific folders

clear; clc; close all;

image_folders = uipickfiles;

for i = 1:length(image_folders)
    cd(image_folders{1,i})
    imds = imageDatastore(image_folders{1,i});
    imgs = readall(imds);
    for ii = 1:length(imds.Files)
        I = imgs{ii,1};
        saveName = strcat('Im_',num2str(ii,'%4.4d'),'.png');
        imwrite(I,saveName)
    end
end


