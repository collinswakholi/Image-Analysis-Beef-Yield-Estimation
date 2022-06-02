function fRank = rank_featuresNCA(cal_s,val_s)

mdl = fsrnca(cal_s(:,2:end),cal_s(:,1),'FitMethod','exact', ...
    'Solver','lbfgs');
L = loss(mdl,val_s(:,2:end),val_s(:,1));
rng(1) % For reproducibility 
n = size(cal_s,1);
cvp = cvpartition(size((cal_s),1),'kfold',5);
numvalidsets = cvp.NumTestSets;
lambdavals = linspace(0,50,20)*std(cal_s(:,1))/n;
lossvals = zeros(length(lambdavals),numvalidsets);

for i = 1:length(lambdavals)
    for k = 1:numvalidsets
        X = cal_s(cvp.training(k),2:end);
        y = cal_s(cvp.training(k),1);
        Xvalid = cal_s(cvp.test(k),2:end);
        yvalid = cal_s(cvp.test(k),1);

        nca = fsrnca(X,y,'FitMethod','exact', ...
             'Solver','minibatch-lbfgs','Lambda',lambdavals(i), ...
             'GradientTolerance',1e-4,'IterationLimit',30);
        
        lossvals(i,k) = loss(nca,Xvalid,yvalid,'LossFunction','mse');
    end
end

meanloss = mean(lossvals,2);
[~,idx] = min(meanloss);
bestlambda = lambdavals(idx);
bestloss = meanloss(idx);

mdl = fsrnca(cal_s(:,2:end),cal_s(:,1),'FitMethod','exact', ...
    'Solver','lbfgs','Lambda',bestlambda);

figure
plot(mdl.FeatureWeights,'ro')
xlabel('Feature Index')
ylabel('Feature Weight')
grid on

fRank = mdl.FeatureWeights;