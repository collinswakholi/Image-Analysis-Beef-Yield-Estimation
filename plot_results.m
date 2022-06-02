function plot_results(Results, units)

cal = Results.cal.real;
val = Results.val.real;

yCal = cal(:,1);
yPred = val(:,1);
predCal = Results.cal.prediction; 
predVal = Results.val.prediction;

Rc2 = Results.performance(1,1);
RMSEc = Results.performance(1,2); 
Rp2 = Results.performance(1,3); 
RMSEp = Results.performance(1,4);

% selec = remove_outrageous(yPred,predVal,RMSEc);
% yPred = yPred(selec);
% predVal = predVal(selec);

if ~exist('units', 'var')
    units = 'kg';
end

if strcmp(units,'kg')
    label = 'weight';
else
    label = 'LMP';
end


figure130 = figure('Color',[1 1 1]);
axes1 = axes('Parent',figure130);

figure130; h1 = scatter(yCal,predCal,200,'DisplayName','Calibration',...
    'MarkerFaceColor',[0.313725501298904 0.313725501298904 0.313725501298904],...
    'MarkerEdgeColor',[0 0 0],...
    'LineWidth',1.5);
    fc = fit(yCal,predCal,'poly1','Robust','Bisquare');
    
hold on;
    h2 = scatter(yPred, predVal,180,'DisplayName','Validation','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[0 0 0],...
    'LineWidth',1.5);
    fp = fit(yPred,predVal,'poly1','Robust','Bisquare');
    
    
aaa = round(ylim.*[0.95,1.05]);
    
    xlim(axes1,[aaa(1) aaa(2)]);
    ylim(axes1,[aaa(1) aaa(2)]);
    

    % Set the remaining axes properties
    set(axes1,'FontSize',16);
    legend1 = legend(axes1,'show');
    set(legend1,...
        'Position',[0.726549597363894 0.131785541599193 0.155773417492578 0.0859621427600693],...
        'EdgeColor',[1 1 1]);

    % Create line
    hold on;
    h3 = plot(fc); hold on; 
    h4 = plot(fp);
    set(h3,'DisplayName','Calibration Fit','Color',[0 0 0],'LineWidth',2.0,'LineStyle','-.');
    set(h4,'DisplayName','Prediction Fit','Color',[1 0 0],'LineWidth',2.0,'LineStyle','-.');

    % Create textbox
    annotation(figure130,'textbox',...
        [0.157509157509158 0.690865154562962 0.163943350558577 0.220820183163562],...
        'String',{strcat('R^2 cal = ',num2str(round(Rc2,2))),strcat('RMSEC = ',num2str(round(RMSEc,2)),units),strcat('R^2 pred = ',num2str(round(Rp2,2))),strcat('RMSEP = ',num2str(round(RMSEp,2)),units)},...
        'FontSize',14,...
        'FontAngle','italic',...
        'EdgeColor',[1 1 1]);
    
    % Create Box
    box(axes1,'on');
    set(axes1,'LineWidth',2,'GridAlpha',0.1,'XGrid','on','YGrid','on');
    
    % Create ylabel
    ylabel(['Predicted ',label,' (',units,')']);

    % Create xlabel
    xlabel(['Actual ',label,' (',units,')']);