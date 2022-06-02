clear;
clc;

%% % read images and show
dataSetDir = 'F:\Collins_ops\Deep_learning\Carcass_outline_detection_MATLAB\resized';
imageDir = fullfile(dataSetDir,'Images');
labelDir = fullfile(dataSetDir,'Labels');

imds = imageDatastore(imageDir);
im_names = imds.Files;
len = length(im_names);

n = 25;
I = readimage(imds,n);

classNames = ["Background","Carcass"];
labelIDs   = [0 1];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
px_names = pxds.Files;

C = readimage(pxds,n);
B = labeloverlay(I,C);
figure; imshow(B)

%% Split the data into training, testing and validation 

train = 0.5; % for training
validate = 0.2; % for validation
test = 0.3; % for testing

rn = randi(len,len,1);
tr_idx = rn(1 : round(train*len));
valid_idx = rn(round(train*len)+1 : round((train+validate)*len));
test_idx = rn(round((train+validate)*len)+1 : end);

im_names_tr = im_names(tr_idx);
px_names_tr = px_names(tr_idx);

im_names_val = im_names(valid_idx);
px_names_val = px_names(valid_idx);

im_names_test = im_names(test_idx);
px_names_test = px_names(test_idx);

imds_train = imageDatastore(im_names_tr);
pxds_train = pixelLabelDatastore(px_names_tr,classNames,labelIDs);

imds_valid = imageDatastore(im_names_val);
pxds_valid = pixelLabelDatastore(px_names_val,classNames,labelIDs);

imds_test = imageDatastore(im_names_test);
pxds_test = pixelLabelDatastore(px_names_test,classNames,labelIDs);

%% UNET network layers (please modify layers as you see fit)
% no need to build network, matlab 2019 has inbuilt UNET network
% Read about UNET network

imageSize = [512 512 3];
numClasses = 2;

choice = 'DeepLab';% 'UNET' or 'DeepLab'

if strcmp(choice,'UNET')
    
    Enc_depth = 4; % default is 4
    fsize = 3;
    
    lgraph = unetLayers(imageSize, numClasses,...
        'EncoderDepth',Enc_depth,...
        'FilterSize',fsize); % you can edit network by editing Lgraph

    % we removed "Segmentation layer", replace it with "dice coeficient layer"
    layer_dice = dicePixelClassificationLayer('Name','dice_layer'); % try 'tversky' loss also, 
    lgraph = replaceLayer(lgraph,'Segmentation-Layer',layer_dice);
    
elseif strcmp(choice,'DeepLab') % using DeepLabV3
    % You must install the corresponding network add-on.
    
    pre_net = 'resnet18'; % 'resnet18', 'resnet50', 'mobilenetv2', 'xception'
    lgraph = deeplabv3plusLayers(imageSize,numClasses,pre_net);
end

analyzeNetwork(lgraph)

%% Augment data and Train network

augmenter = imageDataAugmenter(...
    'RandRotation',[-20 20],...
    'RandXReflection',true,...
    'RandYReflection',true);

ds_training = pixelLabelImageDatastore(imds_train,pxds_train,...
    'DataAugmentation',augmenter);
ds_valid = pixelLabelImageDatastore(imds_valid,pxds_valid,...
    'DataAugmentation',augmenter);

options = trainingOptions(...
    'sgdm', ...
    'InitialLearnRate',1e-3,...
    'Verbose',true,...
    'ValidationData', ds_valid,...
    'ValidationFrequency',10,...
    'validationPatience',5,...
    'MaxEpochs',10,...
    'L2Regularization',0.005,...
    'MiniBatchSize',8,...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.3,...
    'LearnRateDropPeriod',3,...
    'Plots','training-progress',...
    'VerboseFrequency',1);

net = trainNetwork(ds_training,lgraph,options);

save(['my_',choice,'.mat'],'net','options')

clearvars -except net imds_test pxds_test choice
%% predict from image

% load network if doesn't exist

if ~exist('net','var')
    choice = 'DeepLab';% 'UNET' or 'DeepLab'
    load(['my_',choice,'.mat'])
end

save_test_folder = 'test_results';
try
    rmdir(save_test_folder)
end

mkdir(save_test_folder);

I = readimage(imds_test,randi(100));
tic
pxdsPred = semanticseg(imds_test,net,'WriteLocation',save_test_folder);
toc
metrics = evaluateSemanticSegmentation(pxdsPred,pxds_test);

len1 = length(imds_test.Files);

nz = 20;
if nz>len1
    nn = len1;
end
for i = 1:nz
    rr = round(len1*rand);

    I = readimage(imds_test,rr);
    label = readimage(pxds_test,rr);
    I_label = labeloverlay(I,label);

    tic
    [C,scores] = semanticseg(I,net);
    toc
    B = labeloverlay(I, C);

    figure
    imshow(imtile({I,I_label,B},'GridSize', [1 3]),'InitialMagnification',200)
    xlabel (sprintf('Original Image                                        Original Mask                                         Predicted Mask'),...
        'FontSize',20, 'FontWeight', 'bold')
    pause(2)
end
