% Main script
% steps

clear; clc; close all;
%% initial values
ccs = [];
ccs.state = 'Warm'; % Cold, Warm
ccs.side = 'Left'; % Right of Left
ccs.select_image = {'1.','9.','3.','4.'}; % select which images to use
ccs.bs_dims = 1; % use bluescreen dimensions to standardize features
ccs.plt = 0; % show detected features (1,yes, 0 is no)
ccs.deepLearn = 1; % run deep learning

%% deep learning initalize code
if ccs.deepLearn == 1
    [a,aa] = system('C:\ProgramData\Anaconda3\Scripts\activate.bat & F: & cd F:\Collins_ops\Carcass_data_analysis_2020\deep_learning_model & call commands_deep.bat &');
end
%% Read image & mask data store + meta_data
load('meta_data.mat');

mt = cell2table(meta_data);
meta_names = mt.meta_data(:,2);
meta_values = mt.meta_data(:,1);

imds = imageDatastore('out');
mskds = imageDatastore('mask_n');

imNames = imds.Files;
mskNames = mskds.Files;


idx1 = find(contains(imNames,ccs.state,'IgnoreCase',true));
idx2 = find(contains(imNames,ccs.side,'IgnoreCase',true));
idx = intersect(idx1,idx2);

new_imNames = imNames(idx);

Idz = [];
MV = [];
for i = 1:length(new_imNames)
    splt = strsplit(new_imNames{i,1},'\');
    idz = find(contains(mskNames,splt{1,end}));
    idm = find(contains(meta_names,splt{1,end}));
    Idz = [Idz;idz];
    mv = meta_values(idm,:);
    try
        MV = [MV;mv{1,1}];
    catch
        MV = [MV;[mean(mv{1,1}),mean(mv{1,1})]];
    end
end

new_mskNames = mskNames(Idz);
disp([num2str(length(new_imNames)),...
    '_Images_',num2str(round(length(new_imNames)/length(ccs.select_image))),...
    '_Carcasses'])

% sort image numbers
for ni = 1:length(ccs.select_image)
    nz = ccs.select_image{1,ni};
    idn = find(contains(new_imNames,nz));
    eval(['Imgs',nz,'Images = new_imNames(idn);'])
    eval(['Imgs',nz,'Masks = new_mskNames(idn);'])
    eval(['Imgs',nz,'MetaValue = MV(idn,:);'])
end

pause(15)
%% deep learning feature predict

if ccs.deepLearn == 1
disp('+ [Deep learning prediction...]'); tic
    net_in_size = 512;
    BW_deep = {};
    for i = 1:length(Imgs1.Images)
        Img = imread(Imgs1.Images{i,1});
        [Im_dp,II] = deep_predict(Img,net_in_size,1);
%         imshowpair(Img,Im_dp)
        gray = rgb2gray(Im_dp);
        BW = imbinarize(gray,0.83*graythresh(gray));
        BW = imclose(BW,strel('disk',31));
        BW = bwareafilt(BW,1);
        BW = bwmorph(BW,'thin',inf);
        imshowpair(Img,bwmorph(BW,'fatten',2))
        drawnow
        BW_deep{i,1} = BW;
        disp([num2str(i),'/',num2str(length(Imgs1.Images))])
    end
    toc;disp('complete')
    Imgs1.DL_masks = BW_deep;
end

% kill anaconda cmd
!taskkill -f -im cmd.exe
!taskkill -f -im conhost.exe

clearvars -except Imgs1 Imgs9 Imgs3 Imgs4 ccs
%% Load the images, extract features
clc; disp('+ [Reading Images, Extracting features...]'); tic

load('keypoints.mat');

no_images = length(Imgs1.Images);
Name = {};

Feat1 = [];
Feat9 = [];
Feat3 = [];
Feat4 = [];
parfor i = 1:no_images
    name = extract_name(Imgs1.Images{i,1},ccs);
    
    I1 = imread(Imgs1.Images{i,1});
    I9 = imread(Imgs9.Images{i,1});
    I3 = imread(Imgs3.Images{i,1});
    I4 = imread(Imgs4.Images{i,1});
    
    mask1 = imread(Imgs1.Masks{i,1});
    mask9 = imread(Imgs9.Masks{i,1});
    mask3 = imread(Imgs3.Masks{i,1});
    mask4 = imread(Imgs4.Masks{i,1});
    
    II1 = I1;
    II9 = I9;
    II3 = I3;
    II4 = I4;
    
    II1(repmat(~mask1,[1 1 3])) = 0;
    II9(repmat(~mask9,[1 1 3])) = 0;
    II3(repmat(~mask3,[1 1 3])) = 0;
    II4(repmat(~mask4,[1 1 3])) = 0;
    
    if ccs.deepLearn == 1
        mask_dl = Imgs1.DL_masks{i,1};
        feat1 = get_feat1(II1,mask1,ccs,keypoints,mask_dl);
    else
        feat1 = get_feat1(II1,mask1,ccs,keypoints);
    end
    feat9 = get_feat9(II9,mask9,ccs);
    feat3 = get_feat34(II3,mask3,ccs);
    feat4 = get_feat34(II4,mask4,ccs);
    
    Feat1 = [Feat1;feat1];
    Feat9 = [Feat9;feat9];
    Feat3 = [Feat3;feat3];
    Feat4 = [Feat4;feat4];
    
    Name{i} = name;
    disp([num2str(i),'/',num2str(no_images)])
    
end
toc; disp('Done...')

clearvars -except Imgs1 ccs Feat1 Feat9 Feat3 Feat4 Name

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

Feat_coord = [Feat1(:,3),Feat9(:,3),Feat3(:,3),Feat4(:,3)];
Feat_Name = [strcat(Feat1{1,1},'_1'),strcat(Feat9{1,1},'_9'),...
    strcat(Feat3{1,1},'_3'),strcat(Feat4{1,1},'_4')];
file_name = Name';
file_name(remove) = [];


Features = [];
Features.props = ccs;
Features.Feat_values = Feat_combo;
Features.Feat_Coordinates = Feat_coord;
Features.Feat_names = Feat_Name;
Features.File_names = file_name;
Features.Sample_number = length(Feat1);
Features.checkerSize = Imgs1.MetaValue;

t = datetime('today');
dt = datestr(t,28);
if ccs.deepLearn==1
    sav_name = 'DL_';
else
    sav_name = '';
end

im_numbers  = strjoin(ccs.select_image,'');
im_numbers = strjoin(strsplit(im_numbers,'.'),'_');

eval(['save ',ccs.side,'_',ccs.state,'_',im_numbers,sav_name,dt,'.mat Features'])

%%