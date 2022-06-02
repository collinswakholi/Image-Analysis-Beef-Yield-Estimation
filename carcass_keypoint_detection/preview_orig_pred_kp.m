% preview key points original and predicted
clear; clc; close all;

% load image datastore
direc = uigetdir;
imds = imageDatastore(direc);
len = length(imds.Files);

% load keypoints
% orig = readmatrix('original_points.csv');
% orig_2 = orig(2:end,2:end);
pred = readmatrix('predicted_points_ex.csv');
pred_2 = pred(2:end,2:end);
% x_o = []; y_o = [];
x_p = []; y_p = [];
for im = 1:size(pred_2,2)
    if rem(im,2) % odd
%         x_o = [x_o,orig_2(:,im)];
        x_p = [x_p,pred_2(:,im)];
    else
%         y_o = [y_o,orig_2(:,im)];
        y_p = [y_p,pred_2(:,im)];
    end
end

% remove values in the corners
% hyp_orig_out = (hypot(x_o,y_o))>1;
hyp_pred_out = (hypot(x_p,y_p))>200;

% X_o = x_o.*hyp_orig_out;
% Y_o = y_o.*hyp_orig_out;
X_p = x_p.*hyp_pred_out;
Y_p = y_p.*hyp_pred_out;

% error = abs(hypot(X_p,Y_p) - hypot(X_o,Y_o));

num = 30; % number of images to display
r_idx = randi(len,num,1);

for i = 1:num
    figure(i)
    ii = r_idx(i);
    imshow(imds.Files{ii,1});
    
    
%     hold on;
%     scatter(X_o(ii,:),Y_o(ii,:),200,'o',...
%         'LineWidth',1.5,'MarkerEdgeColor', 'k','MarkerFaceColor','y')
    hold on;
    scatter(X_p(ii,:),Y_p(ii,:),250,'*','LineWidth',2,...
        'MarkerEdgeColor', 'g','MarkerFaceColor','b')
end

% regression between original and predicted points
Hyp_pred = hypot(X_p,Y_p);
Hyp_orig = hypot(X_o,Y_o);

% remove outliers
Xx = Hyp_pred(:);
Yy = Hyp_orig(:);
idx_in = find((abs(Xx-Yy))<180);
Yyn = Yy(idx_in);
Xxn = Xx(idx_in);
mdl = fitlm(Xxn,Yyn,'RobustOpts','on');
figure();plot(mdl)