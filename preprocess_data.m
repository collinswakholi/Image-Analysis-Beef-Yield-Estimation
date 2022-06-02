% preprocess features 
clear; clc; close all;

% run this code for every feature mat file below

% im_pro-based features

% name = 'Left_Cold_1_9_3_4_Jul2020.mat';
% name = 'Left_Warm_1_9_3_4_Jul2020.mat';
% name = 'Right_Cold_1_9_3_4_Jul2020.mat';
% name = 'Right_Warm_1_9_3_4_Jul2020.mat';

% im_pro + DL-based features

% name = 'Left_Cold_1_9_3_4_DL_Aug2020.mat';
% name = 'Left_Warm_1_9_3_4_DL_Oct2020.mat';
% name = 'Right_Cold_1_9_3_4_DL_Aug2020.mat';
name = 'Right_Warm_1_9_3_4_DL_Oct2020.mat';

load(name);


f_values = Features.Feat_values;
ch_size = Features.checkerSize;
len = size(f_values,1);

if contains(name,'_DL_')
    load('new_norm.mat')
else
    load('norm_mat.mat')
end

nf_values = abs(f_values)./norm_mat;


ch_size = ch_size(1:len,:);

A_ch = ((ch_size(:,1).*ch_size(:,2)))/(34*34);

Nan = find(isnan(A_ch));
A_ch(Nan) = (31*31)/(34*34);
% divide all variables with A_ch

F_values = nf_values./A_ch;

Data.Raw = F_values;
Data.Mean = Mean_normalize(F_values);
Data.Range = Range_normalize(F_values);
Data.Max = Max_normalize(F_values);
Data.SNV = SNV(F_values);
Data.SG1 = Savitzky_Golay_1st(F_values);

Features.Data = Data;

Features.Sample_number = size(F_values,1);

save(['New_',name],'Features')