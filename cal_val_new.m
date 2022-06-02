clear; close all; clc;

delete('PLS_results.txt');

load('All.mat');

select = 'warm';
k=8; %preprocessing method

if select == 'cold'
    Data = All.warm.xy_data;
else
    Data = All.cold.xy_data;
end

fields = fieldnames(Data);
for i = 1:length(fields)
    disp(strcat(fields{i,1},'..............',num2str(i)));
end
xx = input('\nPlease choose beef to process....\n');

    name = fields{xx,1};
    disp(strcat('"',name,'"...has been selected...'));
   
  %% separate calibration validation prediction sets
  
  calibration = 0.7;
  validation = 0.3;
  prediction = 0.0;
  
    eval(['data1',(strcat('= Data.',name,';'))]);
    
%     data1(:,62:1086)=[];
%     data1 = data1(:,1086:end);
    

    %shuffle
    sz = size(data1,1);
    s_data = data1(randperm(sz), :);
    
    cal = data1(1:round(calibration*sz),:);
    val = data1((1+round(calibration*sz)):round((calibration+validation)*sz),:);
    pred = data1((1+round((calibration+validation)*sz)):end,:);
    
%%
[results,BETA2,val_pred,cal_pred] = plsr_1(cal,val,k);


% rrr = 0;
% while rrr <= 10000
% zz = input('\nPlease input cut off....\n');
%     if zz < 0.01
%         [results,BETA2,val_pred,cal_pred] = plsr_1(cal,val,k);
%     else
%         cal_idx = find(abs(cal(:,1)-cal_pred)<0.8*zz);
%         val_idx = find(abs(val(:,1)-val_pred)<zz);
%         cal1 = cal(cal_idx,:);
%         val1 = val(val_idx,:);
% 
%         [results1,BETA21] = plsr_1(cal1,val1,k);
%         
%         rrr = rrr+1;
%     end
% end
% dlmwrite('PLS_results.txt', results, 'delimiter', '\t', 'newline', 'pc', '-append');

                
                