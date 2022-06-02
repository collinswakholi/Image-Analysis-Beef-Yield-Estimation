function Results = fitlm_model(cal,val,rem_outlier,show)

if ~exist('show', 'var')
    show = 0;
end

    tr_mdl = fitlm(cal(:,2:end),cal(:,1),'RobustOpts','on');%
    
    pred_val = predict(tr_mdl,val(:,2:end));
    pred_cal = predict(tr_mdl,cal(:,2:end));
    
    if rem_outlier
        cal_residuals = cal(:,1)-pred_cal;
        val_residuals = val(:,1)-pred_val;
        
        disp('outlier index')
        idx_cal = find(isoutlier(cal_residuals,'grubbs'))
        idx_val = find(isoutlier(val_residuals,'grubbs'))
        
        cal(idx_cal,:)=[];
        val(idx_val,:)=[];
        
        num_outlier = length([idx_cal;idx_val]);
        tr_mdl = fitlm(cal(:,2:end),cal(:,1),'RobustOpts','on');% ,'RobustOpts','on'
    
        pred_val = predict(tr_mdl,val(:,2:end));
%         pred_cal = predict(tr_mdl,cal(:,2:end));
    else
        num_outlier = [];
    end

    mdl_pr = fitlm(pred_val,val(:,1),'RobustOpts','on');

    Rc2 = tr_mdl.Rsquared.Ordinary;
    Rp2 = mdl_pr.Rsquared.Ordinary;

    RMSEc = tr_mdl.RMSE;
    RMSEp = mdl_pr.RMSE;
    
    Results = [];
    Results.model = tr_mdl;
    Results.cal.real = cal;
    Results.cal.prediction = tr_mdl.Fitted;
    Results.val.real = val;
    Results.val.prediction = pred_val;
    Results.performance = [Rc2,RMSEc,Rp2,RMSEp];
    Result.numOutlier = num_outlier;
    
    
    
if show == 1
    plot_results(Results, '%')
    plot_residuals(Results)
end

