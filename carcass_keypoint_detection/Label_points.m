clear; clc; close all;


im_dir = uigetdir(pwd);
len1 = length(strsplit(im_dir,'\'));

saving_folder = 'Key_points'; %saving folder name
mkdir(saving_folder)

imds = imageDatastore(im_dir);


for i = 650:length(imds.Files)
    im_path = imds.Files{i,1};
    split1 = strsplit(im_path,'\');
    split1{1,len1} = saving_folder;
    savingDir = strjoin(split1,'\');
    
    splt = strsplit(savingDir,'.');
    im_name = strcat(splt{1},'.mat');
    
    I = imread(im_path);
    [~, pts] = Im_points(I,  11, 10);
%     imshowpair(I,Inb,'montage')
%     pause(0.5)
%     imwrite(msk,savingDir);
    save(im_name, 'pts')
    disp(num2str(i))
end