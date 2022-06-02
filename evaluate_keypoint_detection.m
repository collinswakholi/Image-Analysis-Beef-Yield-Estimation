% label random images from left warm carcass
% test accuracy of prediction

% load 20 random images from left carcass

clear; clc; close all;
%% initial values
ccs = [];
ccs.state = "Warm"; % Cold, Warm
ccs.side = "Right"; % Right of Left
ccs.select_image = "Camera1_1";

imds = imageDatastore('out');
mskds = imageDatastore('mask_n');

imNames = imds.Files;
mskNames = mskds.Files;

idx1 = find(contains(imNames,ccs.state,'IgnoreCase',true));
idx2 = find(contains(imNames,ccs.side,'IgnoreCase',true));
idx3 = find(contains(imNames,ccs.select_image,'IgnoreCase',true));
idx = intersect(idx3, intersect(idx1,idx2));

new_imNames = imNames(idx);

nn = 30;
rand_idx = randi(length(idx),nn,1);

load('keypoints.mat')
I_ref = keypoints.WR{1, 1};
mask_out = keypoints.WR{1, 2};
mask_in = keypoints.WR{1, 3};

% load('KeyPoint_eval.mat')
% len = length(Result.Pts);
C = {};
prePts = {};
% se = strel('disk',9);
for i = 1:nn
%     if len<i
        I = imread(new_imNames{i,1});
        splt = strsplit(new_imNames{i,1},'\');
        idz = find(contains(mskNames,splt{1,end}));
        mask = imread(mskNames{idz,1});

        I(repmat(~mask,[1 1 3])) = 0;

        % label image
        pts = get_points(I);
        pts = pts(1:16,:);

        % predict points
        Coords = match_keypoints(I_ref,I,mask_out,mask_in,mask,1,1);
        predict_pts = Coords.output.cc;

        Pts{i} = pts;
        prePts{i} = predict_pts;
    
%     else
%         pts = Result.Pts{i};
%         predict_pts = Result.prePts{i};
%     end
    x_label = hypot(pts(:,1),pts(:,2));
    pred_label = hypot(predict_pts(:,1),predict_pts(:,2));

    [fitobject,gof,output] = fit(pts(:),predict_pts(:),'poly1');
    
    Rsq{i} = gof.rsquare;
    MSE{i} = gof.sse;
    RMSE{i} = gof.rmse;
end

plot_points(I,predict_pts,'b','b',1.5);
hold on;
scatter(predict_pts(:,1),predict_pts(:,2),150,'*','b','LineWidth',1.5); 

Result.mean_Rsq = mean(cell2mat(Rsq));
Result.mean_MSE = mean(cell2mat(MSE));
Result.mean_RMSE = mean(cell2mat(RMSE));

Result.gmean_Rsq = geomean(cell2mat(Rsq));
Result.gmean_MSE = geomean(cell2mat(MSE));
Result.gmean_RMSE = geomean(cell2mat(RMSE));

Result.Rsq = Rsq;
Result.RMSE = RMSE;
Result.Pts = Pts;
Result.prePts = prePts;

save('KeyPoint_eval.mat','Result')
