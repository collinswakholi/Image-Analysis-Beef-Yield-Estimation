% Remove background from images

clear all; clc; close all;

im_dir = uigetdir(pwd);
len1 = length(strsplit(im_dir,'\'));

saving_folder = 'Im_data_nb'; %saving folder name
mkdir(saving_folder)

imds = imageDatastore(im_dir);

for i = 1:length(imds.Files)
    im_name = imds.Files{i,1};
    split1 = strsplit(im_name,'\');
    split1{1,len1} = saving_folder;
    savingDir = strjoin(split1,'\');
   
    I = imread(im_name);
    [~,Inb] = createMask_carcass(I);
%     imshowpair(I,Inb,'montage')
%     pause(0.5)
    imwrite(Inb,savingDir);
    
end