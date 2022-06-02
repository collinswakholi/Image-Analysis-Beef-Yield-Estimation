function plot_residuals(Results)

cal = Results.cal.real;
val = Results.val.real;

cal_pred = Results.cal.prediction; 
val_pred = Results.val.prediction;

cal_residuals = cal(:,1)-cal_pred;
val_residuals = val(:,1)-val_pred;

% find(isoutlier(cal_residuals))
figure100 = figure('Color',[1 1 1]);

axes4 = axes('Parent',figure100);
hold(axes4,'on');
figure100;hh1 = scatter(cal(:,1),cal_residuals,200,'DisplayName','Residuals',...
'MarkerFaceColor',[0.313725501298904 0.313725501298904 0.313725501298904],...
'MarkerEdgeColor',[0 0 0],...
'LineWidth',1.5);

hold on

figure100;hh2 = scatter(val(:,1),val_residuals,200,'DisplayName','Residuals',...
'MarkerFaceColor',[1 0 0],...
'MarkerEdgeColor',[0 0 0],...
'LineWidth',1.5);

xlim ([0.75*min(cal(:,1)) 1.1*max(cal(:,1))]);
ylim([1.5*min(val_residuals) 1.5*max(val_residuals)]);

ylabel('Residual Value');

% Create xlabel
xlabel('Actual weight (kg)');

% Create Box
box(axes4,'on');
set(axes4,'LineWidth',2,'GridAlpha',0.1,'XGrid','on','YGrid','on');

% Set the remaining axes properties
set(axes4,'FontSize',16);

%     legend2 = legend(axes4,'show');

% Create line
annotation(figure100,'line',[0.13125 0.904166666666667],...
[0.457982346832814 0.458982346832814],'LineWidth',2,'LineStyle','--');

