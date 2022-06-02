clear all, close all, clc

delete('Cal_accuracy.txt');
delete('Val_accuracy.txt');

delete('PLS_results.txt');

wavelength = xlsread('New_SWIR_wavelength.xlsx');
wave_range = '2:275';

rowdata1 = xlsread('Calibration.xlsx');
rowdata2 = rowdata1(:,str2num(wave_range));

Prediction_data1 = xlsread('Validation.xlsx');
Prediction_data2 = Prediction_data1(:,str2num(wave_range));
 M = 8;
 rowdata2 = smoothing_mean(rowdata2,M);
 Prediction_data2 = smoothing_mean(Prediction_data2,M);
 
for k = 1:8;
    
    if k==1;
% 1. Mean normalization
        [P_rowdata] = Mean_normalize (rowdata2);     
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = Mean_normalize (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);
         
    elseif k==2;
% 2. Maximum normalization
        [P_rowdata] = Max_normalize (rowdata2);   
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = Max_normalize (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);
         
    elseif k==3;
% 3. Range normalization
        [P_rowdata] = Range_normalize (rowdata2); 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = Range_normalize (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);
         
    elseif k==4;
% 4. MSC
        [P_rowdata] = MSC (rowdata2); 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = MSC (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);

    elseif k==5;
% 5. SNV
        [P_rowdata] = SNV (rowdata2); 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = SNV (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);


    elseif k==6;
% 6. Savitzky_Golay_1st
        [P_rowdata] = Savitzky_Golay_1st (rowdata2); 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = Savitzky_Golay_1st (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);
         
    elseif k==7;
% 7. Savitzky_Golay_2st
        [P_rowdata] = Savitzky_Golay_2nd (rowdata2); 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = Savitzky_Golay_2nd (Prediction_data2);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);

% 8. raw
    elseif k==8;
        [P_rowdata] = rowdata2; 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = Prediction_data2;     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);
       

%9. Smoothing
    elseif k==9;
        [P_rowdata] = smoothing_mean(rowdata2,M); 
         P_rowdata = cat(2,rowdata1(:,1),P_rowdata);
        [Prediction_data3] = smoothing_mean(Prediction_data2,M);     
         Prediction_data3 = cat(2,Prediction_data1(:,1),Prediction_data3);
       
    end

   [Rc2,SEC,Rp2,SEP,PCn,Prediction_value,Prediction_R2,Prediction_SEP,Opimal_PCn1,BETA2,Cal_Prediction_value] = PLS_DA_SIMPLS (P_rowdata,Prediction_data3);
   
%plot for calibration
    figure(k),h1 = gscatter(rowdata1(:,1), Cal_Prediction_value, rowdata1(:,1),'rb','v^',4,'off');
    set(h1,'LineWidth',1.5)
    annotation('textbox', [0.68 0.13 0.21 0.07],'String',{'Baseline = 0.5'},'FontSize',12, 'FontName','Times New Roman','FontWeight', 'bold');
    %(starting point of box, distant from base of the box, inside space, distance from lower
    %line and lower part of writing inside the box)
    xlim([-0.5 1.5]); 
    xlabel('Actual values')
    ylabel('Predicted values')
    legend('Treated','Untreated','Location','NW')
    title('Classification for Calibration')
    
    % plot for validation
 
    figure(11),h1 = gscatter(Prediction_data1(:,1), Prediction_value, Prediction_data1(:,1),'rb','v^',4,'off');
    set(h1,'LineWidth',1)
    annotation('textbox', [0.68 0.13 0.21 0.07],'String',{'Baseline = 0.5'},'FontSize',12, 'FontName','Times New Roman','FontWeight', 'bold');
    %(starting point of box, distant from base of the box, inside space, distance from lower
    %line and lower part of writing inside the box)
    xlim([-0.5 1.5]); 
    xlabel('Actual values')
    ylabel('Predicted values')
    legend('Non-viable','Viable','Location','NW')
    title('Classification for Validation')
   
    
    
    % plot for validation
  figure(k),h1 = gscatter(Prediction_data1(:,1), Prediction_value, Prediction_data1(:,1),'rb','o^',8,'off');
        
    
    [s1 s2] = size(Prediction_data1(:,1));
    
%      test = Numbers of non-viable samples in test (validation)set  
        test =131;
    
         baseline = 0.5;% change baseline according to the sample in calib and valid
                         
            Correct_value = zeros(s1,1);
        
        for ia = 1:test
            if Prediction_value(ia,1) < baseline;
                Correct_value(ia,1) = 1;
            else
                Correct_value(ia,1) = 0;
            end
        end
        
        for ia = test +1 : s1
            if Prediction_value(ia,1) > baseline;
                Correct_value(ia,1) = 1;
            else
                Correct_value(ia,1) = 0;
            end
        end
        
         Non_ct_percent = (sum(Correct_value(1:test)) / (test))*100;
        ct_percent = (sum(Correct_value(test+1:end)) / (s1-test))*100;
        Total_Correct_percent = (sum(Correct_value) / s1)*100;
        
        Num_Non_ct = sum(Correct_value(1:test));
        Num_ct = sum(Correct_value(test+1:end));
        
        results = cat(2,Rc2,SEC,Rp2,SEP,Prediction_R2,Prediction_SEP,PCn);
        
        [s11 s12] = size(rowdata1(:,1));
        
        %cal = Numbers of non-viable samples in calibration set
    cal=307;
    
     baseline1 = 0.5;
        Correct_value1 = zeros(s11,1);
        
        
        for ib = 1:cal
            if Cal_Prediction_value(ib,1) < baseline1;
                Correct_value1(ib,1) = 1;
            else
                Correct_value1(ib,1) = 0;
            end
        end
        
        for ib = cal +1 : s11
            if Cal_Prediction_value(ib,1) > baseline1;
                Correct_value1(ib,1) = 1;
            else
                Correct_value1(ib,1) = 0;
            end
        end
        
        Non_ct_percent1 = (sum(Correct_value1(1:cal)) / (cal))*100;
        ct_percent1 = (sum(Correct_value1(cal+1:end)) / (s11-cal))*100;
        Total_Correct_percent1 = (sum(Correct_value1) / s11)*100;
        
        Num_Non_ct1 = sum(Correct_value1(1:cal));
        Num_ct1 = sum(Correct_value1(cal+1:end));
        
        Cal_results = cat(2,s11, Num_Non_ct1,Num_ct1,Non_ct_percent1,ct_percent1,Total_Correct_percent1);
        Val_results = cat(2,s1, Num_Non_ct,Num_ct,Non_ct_percent,ct_percent,Total_Correct_percent);

        dlmwrite('PLS_results.txt', results, 'delimiter', '\t', 'newline', 'pc', '-append');
        dlmwrite('Cal_accuracy.txt', Cal_results, 'delimiter', '\t', 'newline', 'pc', '-append');
        dlmwrite('Val_accuracy.txt', Val_results, 'delimiter', '\t', 'newline', 'pc', '-append');

        


end
%calibration data plot
Mean_raw = P_rowdata(1:cal,2:end);
Mean_Nonraw = P_rowdata(cal+1:end,2:end);


 figure(10),plot(wavelength(str2num(wave_range)),Mean_raw);

hold on
figure(10),plot(wavelength(str2num(wave_range)),Mean_Nonraw);
xlabel('Wavelength(nm)')
ylabel('Log(1/R)')
xlim([1000 2400]); 


[sr sc] = size(P_rowdata);
     
% Calibaration data mean plot
Mean_Non_ct = mean(Mean_raw,1);
Mean_ct = mean(Mean_Nonraw,1);


figure(11), plot(wavelength(str2num(wave_range)),Mean_Non_ct,'--r','LineWidth',2)
hold on
figure(11), plot(wavelength(str2num(wave_range)),Mean_ct,'b','LineWidth',2)
legend('Treated','Untreated','Location','NW')
% xlim([000 2400])
 xlabel('Wavelength(nm)')
ylabel('Log(1/R)')


beta1 = BETA2(2:end,1);
beta1 = smoothing_mean(beta1',30);
% beta = abs (beta1);  

% plot for Beta coefficient        
figure(12), plot(wavelength(str2num(wave_range)),beta1,'b','LineWidth',1)
legend('Beta coefficients curve of PLS model','ct','Location','NW')
% xlim([1000 2400])
xlabel('Wavelength(nm)')
ylabel('Log(1/R)')

a=zz;
%%

% VIP model 
wavelength = xlsread('FTNIR_wavelength.xlsx');
wavelength1 = wavelength(:,1)';
wave_range = '2:1556';


  data = (P_rowdata(:,2:end));
Y = rowdata1(:,1);  A = PCn;
[T,P,Q,W,R2X,R2Y]=weight(data,Y,A);
 
VIP = vip(data,Y,T,W);
%  
figure(88) % VIP PLOT
plot(wavelength1(str2num(wave_range)),VIP,'b','LineWidth',2)
xlabel('Wavelength(nm)'); ylabel('VIP score'); title('VIP Plot');

%%

% % %% Target projection and selectivity ratio
% %  beta = (beta1)';
% % [t,w,p,sr] = Target_projection(data,beta);
% % %    
% %  figure(8888), plot(wavelength1(str2num(wave_range)),sr,'b','LineWidth',2)
% %  xlabel('Wavelength(nm)');
% % ylabel('Selectivity ratio'); title('Selectivity Ratio Plot');

%% %% SMC

% [smcF smcFcrit SSCregression SSResidual] = smc(beta,data);
% 
% figure(808), plot(wavelength1(str2num(wave_range)),smcF,'b','LineWidth',2)
%  xlabel('Wavelength(nm)');
% ylabel('SMc'); title('SMc Plot');
% % % 
% % % % % %%
   AVIP = find(VIP>0.8); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %  ASR =  find(sr>2); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %  ASMC = find(smcF>50);
% % % % % % %    
   xlswrite('VIP_functional.xlsx',AVIP);
% %  xlswrite('sr_functional.xlsx',ASR);
% %  xlswrite('SMC_functional.xlsx',ASMC);
% % 
% % %  %%
    vipc = rowdata2(:,AVIP);
    vipv = Prediction_data2(:,AVIP);
   wave_vip = wavelength1(:,AVIP);
   vipcal = cat(2,rowdata1(:,1),vipc);
   vipval = cat(2,Prediction_data1(:,1),vipv);
   xlswrite('vip_Cal.xlsx',vipcal);
   xlswrite('vip_Val.xlsx',vipval);
   xlswrite('vip_wave.xlsx',wave_vip');
% %    %% 
% %   src = rowdata2(:,ASR);
% %   srv = Prediction_data2(:,ASR);
% %   wave_sr = wavelength1(:,ASR);
% %   srcal = cat(2,rowdata1(:,1),src);
% %   srval = cat(2,Prediction_data1(:,1),srv);
% %   xlswrite('sr_Cal.xlsx',srcal);
% %   xlswrite('sr_Val.xlsx',srval);
% %   xlswrite('sr_wave.xlsx',wave_sr');
% % %%
% %   smc = rowdata2(:,ASMC);
% %   smv = Prediction_data2(:,ASMC);
% %   wave_smc = wavelength1(:,ASMC);
% %   smccal = cat(2,rowdata1(:,1),smc);
% %   smcval = cat(2,Prediction_data1(:,1),smv);
% %   xlswrite('smc_Cal.xlsx',smccal);
% %   xlswrite('smc_Val.xlsx',smcval);
% %   xlswrite('smc_wave.xlsx',wave_smc');
% % %   
% % % %%
% % Samples = (1:196)';
% % figure(108),scatter(Samples(1:36),t(1:36,1),'r');
% %   hold on;
% % figure(108),scatter(Samples(37:116),t(37:116,1),'g');
% %   hold on;
% % figure(108),scatter(Samples(117:end),t(117:end,1),'b');
% % 




