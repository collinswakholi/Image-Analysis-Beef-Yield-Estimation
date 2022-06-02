function [classes,classCV,score,scoreCV,CVSVMModel,outlierRate,aa,t_aa,u_aa,aa1,t_aa1,u_aa1,beta,bias,sv_indices,n,sensitivity,specificity,sensitivity1,specificity1,svmStruct] = colsvm(training,Prediction,group,pr_group)
%% model training
% svmStruct = svmtrain(training,group);%redifine if need be
svmStruct = fitcsvm(training,group,'ClassNames',[0 1],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',Inf);
    
    ScoreSVMModel = fitSVMPosterior(svmStruct);
         
%model crossvalidation k-fold 10
CVSVMModel = crossval(svmStruct);
[classCV,scoreCV]=kfoldPredict(CVSVMModel);
outlierRate = mean(scoreCV<0);
%model prediction
[classes score]=predict(svmStruct,Prediction);

% classes=svmclassify(svmStruct,Prediction);%redefine if need be


%%
%cp is class performance
cp = classperf(classes, pr_group);

%overall % correct rate
aa=cp.CorrectRate;
% treated % group aa
t_aa=cp.NegativePredictiveValue;
% untreated % group aa
u_aa=cp.PositivePredictiveValue;
%sensitivity
sensitivity=cp.Sensitivity;
% specificity
specificity=cp.Specificity;

%%
%cp is class performance for cross validation
cpv = classperf(classCV, group);

%overall % correct rate
aa1=cpv.CorrectRate;
% treated % group aa
t_aa1=cpv.NegativePredictiveValue;
% untreated % group aa
u_aa1=cpv.PositivePredictiveValue;
%sensitivity
sensitivity1=cpv.Sensitivity;
% specificity
specificity1=cpv.Specificity;
%%

% alpha
beta = svmStruct.Beta;

% Bias
bias = svmStruct.Bias;

% support vector indices
sv_indices = svmStruct.SupportVectorLabels;

% number of support vectors n
S_v = svmStruct.SupportVectors;
sz=size(S_v);
n=sz(1);


