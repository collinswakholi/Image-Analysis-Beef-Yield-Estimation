function [idx_out,idx_in] = get_outliers(Data,out_fr,ncomp,mtd,show)

if ~exist('show','var')
    show = 0;
end

if ~exist('T_fac','var')
    out_fr = 0.01;
end

if ~exist('ncomp','var')
    ncomp = 10;
end

if ~exist('mtd','var')
    mtd = 'pca'; % 'pca', 'pls'
end

Y = Data(:,1);
X = Data(:,2:end);

if strcmp(mtd,'pca')
    [~,score,~,~,var] = pca(X,'Centered',true);
    XS = score(:,1:ncomp);
elseif strcmp(mtd,'pls')
    [~,~,XS] = plsregress(X,Y,ncomp);
end


[~,~,mah,outliers] = robustcov(XS,...
    'OutlierFraction',out_fr, 'ReweightingMethod','rmvn');

idx_out = find(outliers);
idx_in = find(outliers==0);
if show == 1
    d_classical = pdist2(XS, mean(XS),'mahal');
    p = size(XS,2);
    chi2quantile = sqrt(chi2inv(0.999,p));
    
    figure
    plot(d_classical, mah, 'o')
    line([chi2quantile, chi2quantile], [0, 30], 'color', 'r')
    line([0, 6], [chi2quantile, chi2quantile], 'color', 'r')
    hold on
    plot(d_classical(outliers), mah(outliers), 'r+')
    xlabel('Mahalanobis Distance')
    ylabel('Robust Distance')
    title('DD Plot, FMCD method')
    hold off

    d_classical1 = d_classical;
    d1 = mah;

    d_classical1(outliers) = [];
    d1(outliers) = [];

    figure
    plot(d_classical1,d1, 'o')
    line([0 5.5], [0, 5.5])
    xlabel('Mahalanobis Distance')
    ylabel('Robust Distance')
end

% 
% mdl = fitlm(XS,Y, 'RobustOpts','on');
% plotResiduals(mdl,'probability')
% outliers_vals1 = (mdl.Diagnostics.CooksDistance);
% outliers_vals2 = (mdl.Diagnostics.Leverage);

% mean_med = mean([median(outliers_vals),mean(outliers_vals)]);
% outl = outliers_vals > T_fac*mean_med;
% outliers = find(outliers_vals > T_fac*mean_med);