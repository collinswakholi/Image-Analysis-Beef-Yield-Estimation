% Shuffle image data and labels in respective folders
clear,
clc,

dir1 = uigetdir;% for images
dir2 = uigetdir;% for labels
% new_lab_dire = strcat(dir2,'\newLabels');

imds1 = imageDatastore(dir1);
imds2 = imageDatastore(dir2);



% saving_dir = uigetdir;
% before shuffle
for i=1:length(imds2.Files)
    imds1.Files{i,1}
    imds2.Files{i,1}
  
    label = imread(imds2.Files{i,1});
    label1 = round(255*label/2);
    image = imread(imds1.Files{i,1});
    A = imfuse(image, label1, 'blend', 'Scaling', 'joint');
    imshow(A)

    pause(0.5)
%     imwrite(label,label_dir)
%     imwrite(image,img_dir)
end


%%%%%

% load('imageLabelingSession.mat')
% load('gTruth.mat')
% labelNames = table2cell(gTruth.LabelData);
% 
% 
% for i = 1:length(imageLabelingSession.ImageFilenames)
%     I = imread(imageLabelingSession.ImageFilenames{i,1});
%     II = imread(labelNames{i,1});
%     number = num2str(i,'%4.4d');
%     imwrite(I, strcat('new_img\Im_',number,'.png'))
%     imwrite(II, strcat('new_lab\L_',number,'.png'))
%     imshowpair(I,II)
%     pause(0.1)
% end
