% preview keypoint
clear; clc; close all;

im_dir = uigetdir(pwd);
key_point_dir = uigetdir(pwd);

imds = imageDatastore(im_dir);
len = length(imds);
i = 1;

while i>0
   
    Im = imread(imds.Files{i,1});
    splt1 = strsplit(imds.Files{i,1},'\');
    splt2 = strsplit(splt1{1,end},'.');
    kp = load([key_point_dir,'\',splt2{1,1},'.mat']);
    kpts = kp.pts;
    
    idx = find(kpts(:,1)==1);
    
    cla;
    
    imshow(Im);
    hold on;
    scatter(kpts(:,2),kpts(:,3),100,'o','LineWidth',0.6,'MarkerEdgeColor', 'k','MarkerFaceColor','g')
    for ii = 1:length(idx)
        text(kpts(idx(ii),2)+5,kpts(idx(ii),3)+5, num2str(idx(ii)), 'FontSize',28);
    end
    
    xx = input('[1] Do you want to see next image? OR \n[0] Quit\n');
    if xx == 0
        break
    else
        i = i+1;
    end
end