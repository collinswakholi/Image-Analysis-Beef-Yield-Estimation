function pred_result = actual_pred_stats(actual,pred,show,cret)

if  ~exist('show','var')
    show = 1;
end

if  ~exist('cret','var')
    cret = 0;
end

sz = length(actual);

assert(sz==length(pred),'Input vectors must be of same length')

%% kick out outliers
mdl = fitlm(actual,pred);

mn = 0.95*min(min([actual,pred]));
mx = 1.0526*max(max([actual,pred]));

std1 = std(actual);
mn1 = mean(actual);


% outlier, observation with 3x the mean cools distance is an outlier
mdCook= mdl.Diagnostics.CooksDistance;
mdDffits= mdl.Diagnostics.Dffits;
dddist = hypot(mdCook,mdDffits);
% idx = dddist>(graythresh(dddist));

% idx = (mdCook>=2.5*(mean(mdCook)));

if cret == 1
    % find points greater than twice the standard deviation from the mean
    idx = abs(pred-actual)>(2.5*std1);

elseif cret == 2
    % or find points outside the 95th percentile
    idx = abs(pred-actual)>(prctile(abs(pred-actual),95));
elseif cret == 3
    % or find points outside the 95th percentile
    mdCook= mdl.Diagnostics.CooksDistance;
    idx = (mdCook>=2.5*(mean(mdCook)));
else
    idx = [];
end

outliers = [actual(idx==1),pred(idx==1)];

disp([num2str(sum(idx)),'_outliers data points removed'])
actual(idx,:)=[];
pred(idx,:)=[];

%% params

mdl = fitlm(actual,pred);

% Bias 
bias = (sum(actual-pred))/sz;

% MSE
% MSE = (sum((actual-pred).^2))/sz;
MSE = mdl.MSE;

% PRESS
PRESS = sz*MSE;

% SEP
SE = sqrt(MSE-(bias^2));
% SE = sqrt((sum((actual-pred-bias).^2))/(sz-1));

% RMSE
% RMSE = sqrt((sum((actual-pred).^2))/sz);
RMSE = mdl.RMSE;

% R_squared
% R_sq = rsquare(actual,pred);
R_sq = mdl.Rsquared.Ordinary;

pred_result = [];
pred_result.bias = bias;
pred_result.MSE = MSE;
pred_result.PRESS = PRESS;
pred_result.SE = SE;
pred_result.RMSE = RMSE;
pred_result.R_sq = R_sq;

%% plot
if show ==1
    
    figure177 = figure('Color',[1 1 1]);

    % Create axes
    axes177 = axes('Parent',figure177);
    hold(axes177,'on');

    % Create scatter
    scatter(actual,pred,200,'DisplayName','Inliers','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[0 1 0],...
    'LineWidth',1);
    
    if length(outliers)>0
    % Create scatter
    scatter(outliers(:,1),outliers(:,2),300,'DisplayName','Outliers','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[0 0 0],...
    'Marker','hexagram',...
    'LineWidth',1);
    end
    
    xlim(axes177,[mn mx]);
    ylim(axes177,[mn mx]);
    
    % Create ylabel
    ylabel('Predicted weight (kg)','FontAngle','italic');

    % Create xlabel
    xlabel('Actual weight (kg)','FontAngle','italic');

    % Create title
    title('Prediction Plot');

    % Uncomment the following line to preserve the X-limits of the axes
    % xlim(axes1,[9 12]);
    % Uncomment the following line to preserve the Y-limits of the axes
    % ylim(axes1,[9 12]);
    box(axes177,'on');
    % Set the remaining axes properties
    set(axes177,'FontSize',16,'XGrid','on','YGrid','on');
    % Create legend
    legend1 = legend(axes177,'show');
    set(legend1,...
    'Position',[0.710266324310624 0.181148376255258 0.152061853096965 0.106961379842303],...
    'FontAngle','italic',...
    'EdgeColor',[1 1 1]);

    % Create line
    annotation(figure177,'line',[0.131443298969072 0.903350515463918],...
    [0.111280487804878 0.920731707317073],'LineWidth',2,'LineStyle','--');

    % Create textbox
    annotation(figure177,'textbox',...
    [0.177546391752577 0.730182926829269 0.259309278350515 0.160060975609756],...
    'String',{['R^2_p = ',num2str(round(R_sq,2))],['RMSEP = ',num2str(round(RMSE,2)),' kg']},...
    'FontSize',14,...
    'FontAngle','italic',...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);

end
