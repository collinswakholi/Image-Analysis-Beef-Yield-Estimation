function [movingRegistered,tform] = fit_mask (I,Iref,Iter,init_rad,show)
% NOTE: I and Iref should be gray or binary

I = double(I);
Iref = double(Iref);

if nargin < 3
    Iter = 100;
    init_rad = 0.0009;
    show = 0;
end

if show == 1
    figure(1)
    imshowpair(Iref, I,'Scaling','joint');
    title('Before fitting')
end
[optimizer, metric] = imregconfig('multimodal');

optimizer.InitialRadius = init_rad;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = Iter;

tform = imregtform(Iref, I, 'affine', optimizer, metric);
movingRegistered = imwarp(Iref,tform,'OutputView',imref2d(size(Iref)));

if show == 1
    figure(2)
    imshowpair(I, movingRegistered,'Scaling','joint')
    title('After fitting')
end