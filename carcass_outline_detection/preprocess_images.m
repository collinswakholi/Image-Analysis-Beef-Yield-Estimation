% resize images and make them square

clear;
clc;

saving_folder = 'F:\Collins_ops\Deep_learning\Carcass_outline_detection_MATLAB\resized';
if ~exist(saving_folder,'dir')
    mkdir([saving_folder,'\Images']);
    mkdir([saving_folder,'\Labels']);
end

image_dir = 'F:\Collins_ops\Deep_learning\Carcass_outline_detection_MATLAB\Img';
label_dir = 'F:\Collins_ops\Deep_learning\Carcass_outline_detection_MATLAB\mask_n';

imds = imageDatastore(image_dir);
labds = imageDatastore(label_dir);
im_files = imds.Files;
lab_files = labds.Files;

len = length(im_files);


out_size = 512;
fmt = '.png';

for i = 1:len
    name = im_files{i,1};
    splt = strsplit(name,'\');
    im_name = splt{1,end};
    splt2 = strsplit(im_name,'.');
    name_find = splt2{1,1};
    
    lab_idx = find(contains(lab_files,name_find));
    lab_name = lab_files{lab_idx,1};
    
    I = imread(name);
    Lab = imread(lab_name);
    
    try
        Lab = rgb2gray(Lab);
    end
    
%     Ir = imresize(I,[out_size,out_size]);
%     Ilab = imresize(Lab,[out_size,out_size]);
    
    [Ir,Ilab] = square_img_new(I,Lab,out_size,1);
    
%     preview images
%     figure(1); imshowpair(Ir,Ilab,'blend');
    
    imwrite(Ir,[saving_folder,'\Images\',name_find,fmt]); % save image file
    imwrite(Ilab,[saving_folder,'\Labels\',name_find,fmt]); % save image file
    
end
disp('done');