% Main script
% steps

clear; clc; close all;
%% initial values
ccs = [];
ccs.state = 'Cold'; % Cold, Warm
ccs.side = 'Right'; % Right of Left
% ccs.select_image = {'1.'};
ccs.select_image = {'1.','9.','3.','4.'}; % select which images to use
% ccs.select_image = {'1.','3.'}; % select which images to use
ccs.deepLearn = 0; % use deep learning, 1-yes, 0-only use image processing
ccs.bs_dims = 1; % use bluescreen dimensions to standardize features
ccs.plt = 0; % show detected features (1,yes, 0 is no)
%% deep learning initalize code
if ccs.deepLearn == 1
    [a,aa] = system('C:\ProgramData\Anaconda3\Scripts\activate.bat & F: & cd F:\Collins_ops\Carcass_data_analysis_2019\deep_learning_model & call commands_deep.bat &');
end
%% Read image data store and prediction values
imds = imageDatastore('out');
imNames = imds.Files;

selected_im = [ccs.state,ccs.side,ccs.select_image];
idx1 = [];
for i=1:length(selected_im)
    idx = contains(imNames,selected_im{1,i},'IgnoreCase',true);
    idx1 = [idx1,idx];
end
idx = (sum(idx1,2)>2);
new_imNames = imNames(idx == 1);
imds_selec = imds;
imds_selec.Files = new_imNames;


disp([num2str(length(new_imNames)),'_Images_',num2str(round(length(new_imNames)/length(ccs.select_image))),'_Carcasses'])


%% Load the images
disp('+ [Reading Images]'); tic
idx1 = contains(new_imNames,'_1.');
imNames_1 = new_imNames(idx1==1);
imds1 = imds_selec;
imds1.Files = imNames_1;

if sum(contains(ccs.select_image,'9.'))>0.5
    idx9 = contains(new_imNames,'_9.');
    imNames_9 = new_imNames(idx9==1);
else
    imNames_9 = {};
end

if sum(contains(ccs.select_image,'3.'))>0.5
    idx3 = contains(new_imNames,'_3.');
    imNames_3 = new_imNames(idx3==1);
else
    imNames_3 = {};
end
    
if sum(contains(ccs.select_image,'4.'))>0.5
    idx4 = contains(new_imNames,'_4.');
    imNames_4 = new_imNames(idx4==1);
else
    imNames_4 = {};
end

% tic
% Imgs  = gpuArray(readImages(imNames_1,imNames_9,imNames_3,imNames_4,ccs));
% toc

Im_934 = {imNames_9,imNames_3,imNames_4};

[Imgs_1,Imgs_9,Imgs_3,Imgs_4] = deal([]);
Name = {};
k=0;
for i = 1:length(imNames_1)
    name = extract_name(imNames_1{i,1},ccs);
    [go,pos] = checklist4_Im(Im_934,name,ccs);
    I_1 = imread(imNames_1{i,1});

    if go == 1
        I_9 = imread(imNames_9{pos(1),1});
        I_3 = imread(imNames_3{pos(2),1});
        I_4 = imread(imNames_4{pos(3),1});
        k=k+1;
    else
%         [I_1, I_9, I_3, I_4] = deal([]);
        name = [];
    end
    Name = [Name;name]; 
    Imgs_1{k,1} = I_1;
    Imgs_9{k,1} = I_9;
    Imgs_3{k,1} = I_3;
    Imgs_4{k,1} = I_4;
    disp([num2str(i),'/',num2str(length(imNames_1))])
end

toc; disp('Done...')
clearvars -except Name Imgs_1 Imgs_9 Imgs_3 Imgs_4 ccs aa a
%% deep learning feature predict

if ccs.deepLearn == 1
disp('+ [Deep learning prediction...]'); tic
    net_in_size = 512;
    BW_deep = {};
    for i = 1:length(Imgs_1)
        Img = Imgs_1{i,1};
        [Im_dp,II] = deep_predict(Img,net_in_size,1);
%         imshowpair(Img,Im_dp)
        gray = rgb2gray(Im_dp);
        BW = imbinarize(gray,0.83*graythresh(gray));
        BW = imclose(BW,strel('disk',3));
        BW = bwareafilt(BW,1);
        BW = bwmorph(BW,'thin',inf);
        imshowpair(Img,bwmorph(BW,'fatten',2))
        drawnow
        BW_deep{i,1}=BW;
        disp([num2str(i),'/',num2str(length(Imgs_1))])
    end
    toc;disp('complete')
end

% kill anaconda cmd
!taskkill -f -im cmd.exe
!taskkill -f -im conhost.exe
%% extract background and bluescreen
if ccs.plt ==1
    Img1 = 2*imfuse(Imgs_1{end,1},bwmorph(BW_deep{end,1},'fatten',2),'blend');
    imshowpair(Imgs_1{end,1},Img1,'montage')
    for nn = 1:length(Imgs_1)
    %     [mask,image, bs,bb] = createMask_final(II_1{nn,1});
        Img1 = 2*imfuse(Imgs_1{nn,1},bwmorph(BW_deep{nn,1},'fatten',2),'blend');
        figure(2);
        imshowpair(Imgs_1{nn,1},Img1,'montage')
    %     imshow(II_1{nn,1}); hold on; rectangle('Position',bb,'EdgeColor','r')
    end
end
%% process images
% Image 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% get
% features from image 1
disp('+ [Extracting features from Image 1]'); tic
if ccs.deepLearn == 1
    Feat1 = [];
    for i = 1:length(Imgs_1)
        Img = Imgs_1{i,1};
        Deep_msk = BW_deep{i,1};
        [mask,image, bs,bb] = createMask_final(Img);
        feat1 = get_feat1(Img,mask,Deep_msk,ccs);
%         imshow(image); hold on; rectangle('Position',bb,'EdgeColor','r')
        Feat1 = [Feat1;feat1];
        disp([num2str(i),'/',num2str(length(Imgs_1))])
    end

    ccs.plt = 1;
    get_feat1(Img,mask,Deep_msk,ccs);
    ccs.plt = 0;

elseif ccs.deepLearn == 0
    Feat1 = [];
    for i = 1:length(Imgs_1)
        Img = Imgs_1{i,1};
        [mask,image, bs,bb] = createMask_final(Img);
        feat1 = get_feat1(Img,mask,[],ccs);
%         imshow(image); hold on; rectangle('Position',bb,'EdgeColor','r')
        Feat1 = [Feat1;feat1];
        disp([num2str(i),'/',num2str(length(Imgs_1))])
    end
    
    ccs.plt = 1;
    get_feat1(Img,mask,ccs);
    drawnow
    ccs.plt = 0;
    
else
    Feat1 = [];
end
toc; disp('Done...')

% Image 9%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('+ [Extracting features from Image 9]'); tic
if sum(contains(ccs.select_image,'9.'))>0.5
    Feat9 = {};
    for i = 1:length(Imgs_9)
            Img = Imgs_9{i,1};
            [mask,bb] = im9_mask(Img);
            feat9 = get_feat9(Img,mask,ccs,bb);
            Feat9 = [Feat9;feat9];
            disp([num2str(i),'/',num2str(length(Imgs_9))])
    end
    ccs.plt = 1;
    get_feat9(Img,mask,ccs,bb);
    drawnow
    ccs.plt = 0;
else
  Feat9 = {};
end
toc; disp('Done...')

% Image 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('+ [Extracting features from Image 3]'); tic
if sum(contains(ccs.select_image,'3.'))>0.5
    Feat3 = [];
    for i = 1:length(Imgs_3)
            Img = Imgs_3{i,1};
            mask = im3_mask(Img);
            feat3 = get_feat3(Img,mask,ccs);
    %         imshow(image); hold on; rectangle('Position',bb,'EdgeColor','r')
            Feat3= [Feat3;feat3];
            disp([num2str(i),'/',num2str(length(Imgs_3))])
    end
    ccs.plt = 1;
    get_feat3(Img,mask,ccs);
    drawnow
    ccs.plt = 0;
else
    Feat3 = {};
end
toc; disp('Done...')

% Image 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('+ [Extracting features from Image 4]'); tic
if sum(contains(ccs.select_image,'4.'))>0.5
    Feat4 = [];
    for i = 1:length(Imgs_4)
            Img = Imgs_4{i,1};
            mask = im4_mask(Img);
            feat4 = get_feat4(Img,mask,ccs);
    %         imshow(image); hold on; rectangle('Position',bb,'EdgeColor','r')
            Feat4= [Feat4;feat4];
            disp([num2str(i),'/',num2str(length(Imgs_4))])
    end
    ccs.plt = 1;
    get_feat4(Img,mask,ccs);
    drawnow
    ccs.plt = 0;
else
    Feat4 = {};
end
toc; disp('Done...')

%% save extracted features
Feat_combo = [];
remove = [];
for i=1:length(Feat1)
    try
    Feat_combo = [Feat_combo;[Feat1{i,2},Feat9{i,2},Feat3{i,2},Feat4{i,2}]];
    catch
        remove = [remove,i];
    end
end
Feat_Name = [strcat(Feat1{1,1},'_1'),strcat(Feat9{1,1},'_9'),strcat(Feat3{1,1},'_3'),strcat(Feat4{1,1},'_4')];
file_name = Name;
file_name(remove) = [];

Features = [];
Features.props = ccs;
Features.Feat_values = Feat_combo;
Features.Feat_names = Feat_Name;
Features.File_names = file_name;
Features.Sample_number = length(Feat1);

if ccs.deepLearn==1
    sav_name = 'DL';
else
    sav_name = '';
end

im_numbers  = strjoin(ccs.select_image,'');
im_numbers = strjoin(strsplit(im_numbers,'.'),'_');

eval(['save ',ccs.side,'_',ccs.state,'_',im_numbers,sav_name,'.mat Features'])

%%