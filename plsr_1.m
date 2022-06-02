% clear all, close all, clc

function [results,BETA2,Prediction_value,Cal_Prediction_value,PCn]  = plsr_1(Xcal,Xpred,k1,PCn,plt)

if isfile('PLS_results.txt')
    delete('PLS_results.txt');
end

if ~exist('plt','var')
    plt = 0;
end

rowdata2 = Xcal(:,2:end);

Prediction_data2 = Xpred(:,2:end);

for k = k1(1):k1(end)
    
    if k==1
% 1. Mean normalization
        [P_rowdata] = Mean_normalize (rowdata2);     
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = Mean_normalize (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);
         
    elseif k==2
% 2. Maximum normalization
        [P_rowdata] = Max_normalize (rowdata2); 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = Max_normalize (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);
         
    elseif k==3
% 3. Range normalization
        [P_rowdata] = Range_normalize (rowdata2); 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = Range_normalize (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);
         
    elseif k==4
% 4. MSC
        [P_rowdata] = MSC (rowdata2); 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = MSC (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);

    elseif k==5
% 5. SNV
        [P_rowdata] = SNV (rowdata2); 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = SNV (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);


    elseif k==6
% 6. Savitzky_Golay_1st
        [P_rowdata] = Savitzky_Golay_1st (rowdata2); 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = Savitzky_Golay_1st (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);
         
    elseif k==7
% 7. Savitzky_Golay_2st
        [P_rowdata] = Savitzky_Golay_2nd (rowdata2); 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = Savitzky_Golay_2nd (Prediction_data2);     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);

% 8. raw
    elseif k==8
        [P_rowdata] = rowdata2; 
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = Prediction_data2;     
         Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);
 
% % 9 smoothing     
%Basic smoothing size = 5; 
    elseif k==9
         Msize = 5;
         [P_rowdata] = smoothing_mean (rowdata2,Msize);
         P_rowdata = cat(2,Xcal(:,1),P_rowdata);
        [Prediction_data3] = smoothing_mean (Prediction_data2,Msize); 
        Prediction_data3 = cat(2,Xpred(:,1),Prediction_data3);
    end

    
        [Rc2,SEC,Rp2,SEP,PCn,Prediction_value,Prediction_R2,Prediction_SEP,Opimal_PCn1,BETA2,Cal_Prediction_value,RMSEC,RMSEP,Prediction_RMSEP] = PLS_DA_SIMPLS (P_rowdata,Prediction_data3,PCn);

        % plot for calibration;
        
 if plt == 1
   axes1 = axes('Parent',figure(k));
   figure(k),h1 = scatter(Xcal(:,1),Cal_Prediction_value,200,'DisplayName','Calibration',...
    'MarkerFaceColor',[0.313725501298904 0.313725501298904 0.313725501298904],...
    'MarkerEdgeColor',[0 0 0],...
    'LineWidth',1.5);
   %figure(k),h1 = gscatter(Prediction_data1(:,1),Prediction_value,Prediction_data1(:,1));

    % plot for validation
    hold on;
    h2 = scatter(Xpred(:,1), Prediction_value,200,'DisplayName','Validation','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[0 0 0],...
    'LineWidth',1.5);
    
    aaa = ylim;
    
    xlim(axes1,[aaa(1) aaa(2)]);
    % Create ylabel
    ylabel('Predicted weight (kg)');

    % Create xlabel
    xlabel('Actual weight (kg)');

    box(axes1,'on');
    % Set the remaining axes properties
    set(axes1,'FontSize',16);
    legend1 = legend(axes1,'show');
    set(legend1,...
        'Position',[0.726549597363894 0.131785541599193 0.155773417492578 0.0859621427600693],...
        'EdgeColor',[1 1 1]);

    % Create line
    annotation(figure(k),'line',[0.128205128205128 0.902319902319902],...
        [0.109378912685338 0.920922570016474],'LineWidth',2,'LineStyle','--');

    % Create textbox
    annotation(figure(k),'textbox',...
        [0.157509157509158 0.690865154562962 0.163943350558577 0.220820183163562],...
        'String',{strcat('R^2 cal = ',num2str(round(Rc2,2))),strcat('SEC = ',num2str(round(SEC,2)),' kg'),strcat('R^2 pred = ',num2str(round(Rp2,2))),strcat('SEP = ',num2str(round(SEP,2)),' kg')},...
        'FontSize',14,...
        'FontAngle','italic',...
        'EdgeColor',[1 1 1]);

    % Create rectangle
    annotation(figure(k),'rectangle',...
        [0.131647130647131 0.112026359143328 0.774335775335775 0.812191103789128],...
        'LineWidth',2);
    
 end  
    
    % for Calibration
   
%     [s11 s12] =size(P_rowdata);
    
results = cat(2,Rc2,SEC,Rp2,SEP,Prediction_R2,Prediction_RMSEP,PCn);
 dlmwrite('PLS_results.txt', results, 'delimiter', '\t', 'newline', 'pc', '-append');
 

end


% BETA plot

beta = BETA2(2:end,1)';

% figure(14), plot(beta,'b','LineWidth',2)
% %legend('Beta coefficients curve of PLS model','ct','Location','NW')
% ylabel('Beta coefficients curve of PLS model')

% selectivity ratio
% [t,w,p,sr] = Target_projection(P_rowdata,beta);
% 
% figure(18), plot(wavelength1(str2num(wave_range)),sr,'b','LineWidth',2)


