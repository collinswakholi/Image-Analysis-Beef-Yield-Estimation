clear all
clc
close all;

delete('All_Calibration_new.xlsx','All_Validation_new.xlsx')

cal_delete = [40 33 61 57 3 38 17 115 67 123 97 82 69 77];
val_delete = [21 17 3 25 19 29 50 44 38 34];

cal = xlsread('All_Calibration8680.xlsx');
val = xlsread('All_Validation8680.xlsx');

cal(cal_delete,:)=[];
val(val_delete,:)=[];

n1=0.5*size(cal,1);
n2=0.5*size(val,1);

non_viable =  [cal(1:n1,:);val(1:n2,:)];
viable = [cal(n1+1:end,:);val(n2+1:end,:)];

r =0.7;
z = round(r*size(viable,1));

     calibration = cat(1,non_viable(1:z,:),viable(1:z,:));
     validation = cat(1,non_viable(z+1:end,:),viable(z+1:end,:));
%  calibration = cat(1,S_n_v(1:517,:),S_viable(1:586,:));
%  validation = cat(1,S_n_v(518:end,:),S_viable(587:end,:));
% 
% 
 xlswrite('All_Calibration_new.xlsx',calibration);
 xlswrite('All_Validation_new.xlsx',validation);