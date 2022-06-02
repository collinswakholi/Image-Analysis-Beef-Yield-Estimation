% read masks, find corresponding images, make them square size (512 xx 512)
clear
close all;
clc

mask_folder = 'dl_mask_test\masks';
im_folder = 'out';
im_sav_folder = 'dl_mask_test\images';

imds = imageDatastore(im_folder);
im_files = imds.Files;

imds_msk = imageDatastore(mask_folder);
msk_files = imds_msk.Files;
len = length(msk_files);

for i = 1:len
    msk_spt = strsplit(msk_files{i,1},'\');
    mask_name = msk_spt{end};
    idx = find(contains(im_files,mask_name));
    try
        im_name = im_files{idx};

        mask = imread(msk_files{i,1});
        I = imread(im_name);
        sz = size(mask);

        [II,mm] = square_img_new(I,mask,512,1);

        imshowpair(II,mm)
        drawnow
        imwrite(II,[im_sav_folder,'\',mask_name])
        imwrite(mm,msk_files{i,1});
    catch
        delete(msk_files{i,1})
        disp(['image ', num2str(i),' not found']);
    end
end