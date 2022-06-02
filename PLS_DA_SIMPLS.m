
function [Rc2,SEC,Rp2,SEP,PCn,Prediction_value,Prediction_R2,Prediction_SEP,Opimal_PCn1,BETA2,Cal_Prediction_value,RMSEC,RMSEP,Prediction_RMSEP] = PLS_DA_SIMPLS (rowdata,Prediction_data1,PCn)

[sr sc] = size(rowdata);

if ~exist('PCn','var')
    PCn = 15;
end
% wavelength = rowdata(1,2:sc);
RX = rowdata(:,2:sc);
RY = rowdata(:, 1);
TMean_RY = mean(RY);

% Prediction_data_load
P_RX = Prediction_data1(:,2:sc);
P_RY = Prediction_data1(:,1);

% Input the PC number
%     PCn = 15;
    V_Y_Prediction1 = zeros(sr,1);

% Finding the optimal PCn(cross-validation data (leave-out))
vr = sr;
% vr = 5;% n-fold cross-val
[~,~,~,~,~,PCTVAR1,MSE2] = plsregress(RX,RY,PCn,'cv',vr);


% Optimal PC number

Opimal_PCn1 = MSE2(2,:)';
Toptimal_V = sum(Opimal_PCn1);
Opimal_PCn2 = cat(1,Toptimal_V,Opimal_PCn1(1:end-1,1));
Opimal_PCn3 = (Opimal_PCn2-Opimal_PCn1)./Toptimal_V;

[row,~] = find(Opimal_PCn1==min(Opimal_PCn1));
% T_Sum_optimal_value = sum(Opimal_PCn3(1:row,1));

if row > PCn
    row = row -1;
end

for iii = 1:row
      if abs(Opimal_PCn3(iii,1)) <= 0.005
         if sum(PCTVAR1(2,1:iii)) >= 0.85
             break
         end
      end
end 

PCn = iii;

% Find validation data predicted using optimal PCn
for i=1:sr
    Validation_data = RX(i,:);

    if i==1;
        Traing_data = RX(i+1:sr,:);
        Traing_data_RY = RY(i+1:sr,:);
    elseif i==sr;
        Traing_data = RX(1:sr-1,:);
        Traing_data_RY = RY(1:sr-1,:);
    else 
        Traing_data = cat(1,RX(1:i-1,:),RX(i+1:sr,:));
        Traing_data_RY = cat(1,RY(1:i-1,:),RY(i+1:sr,:));
    end
        [~,~,~,~,BETA1] = plsregress(Traing_data,Traing_data_RY,PCn);

% Predicting of Y values
V_Y_Prediction1(i,1) = [ones(1) Validation_data]*BETA1;

end

% Calculate Rp2, RMSEP, SEP
V_Y_Prediction2 = [ones(sr,1) V_Y_Prediction1];
[~,~,~,~,stats1] = regress(RY,V_Y_Prediction2);
Rp2 = stats1(1,1);

% Rp2 = 1-(1-Rp21)*((sr-1)/(sr-PCn-1));
RMSEP = ((sum((V_Y_Prediction1-RY).^2))/(sr))^.5;
P_bias = abs((sum(V_Y_Prediction1-RY))/sr);
SEP = ((sum((V_Y_Prediction1-RY-P_bias).^2))/(sr-1))^.5;
% P_value = V_Y_Prediction1(:,PCn);

size_Pv = size(P_RX);

% The default is 'resubstitution'.
[~,YL,XS,~,BETA2,PCTVAR] = plsregress(RX,RY,PCn);

% % Calculate Rc2, RMSEC, SEC
C_Y_Prediction1 = XS*YL'+TMean_RY;
% RMSEC = Y estimated variance ^.5
Rc2 = PCTVAR(2,:);
Rc2 = sum(Rc2);
RMSEC = MSE2(2,PCn+1)^.5;
SEC = ((sum((C_Y_Prediction1-RY).^2))/(sr-1))^.5;

% Prediction value Using the PLS equation
% Predicting of Y values using separated prediction data
% Prediction_value = [ones(1) P_RX]*BETA2;

Prediction_value = zeros(size_Pv(1),1);
for ia =1:size_Pv(1)
    Prediction_value(ia,:) = [ones(1) P_RX(ia,:)]*BETA2;
end

% Calibration set : Accuracy calculating
Cal_Prediction_value = [ones(sr,1) RX]*BETA2;

% Calculate Prediction_R2, SEC
Prediction_value1 = [ones(size_Pv(1),1) Prediction_value(:,:)];
[~,~,~,~,stats2] = regress(P_RY,Prediction_value1);
Prediction_R2 = stats2(1,1);
% TSS = sum((Prediction_value-mean(Prediction_value)).^2);
% RSS = sum((Prediction_value-P_RY).^2);
% Prediction_R2 = 1 - RSS/TSS;


Prediction_P_bias = ((sum((Prediction_value-P_RY)))/size_Pv(1));
Prediction_SEP = (sum((Prediction_value-P_RY-Prediction_P_bias).^2)/(size_Pv(1)-1))^.5;
Prediction_RMSEP = (sum((Prediction_value-P_RY).^2)/(size_Pv(1)))^.5;
