function [BW,maskedRGBImage,sq_dims] = find_remove_checkers(RGB,Rside,show)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 25-May-2020
%------------------------------------------------------

if ~exist('show', 'var')
    show = 0;
end

if ~exist('Rside', 'var')
    Rside = [0,0];
end

% Convert RGB image to chosen color space
sz = size(RGB);

RGB1 = RGB(:,(round(sz(2)/2):end),:);
I = rgb2lab(RGB1);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 45;
channel1Max = 100;

% Define thresholds for channel 2 based on histogram settings
channel2Min = -60;
channel2Max = 70;

% Define thresholds for channel 3 based on histogram settings
channel3Min = -20;
channel3Max = 70;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;
BW = imclose(BW, strel('disk',3));
BW = bwareaopen(BW,100);
BWw = bwareaopen(BW,1300);
BW = xor(BW, BWw);

reg_props = regionprops(BW,'BoundingBox','PixelIdxList', 'Area');

BB = [];
for i = 1:length(reg_props)
    bb = reg_props(i).BoundingBox;
    BB = [BB;bb];
end
Area = ([reg_props.Area])';

[N,edges] = histcounts(BB(:,1));
[~,mx] = max(N);

try
    lim = [edges(mx-1),edges(mx+1)];
catch
    lim = [edges(mx),edges(mx+2)];
end

fx2 = find((abs(BB(:,3)-BB(:,4))<10)&...
        ((BB(:,1)>lim(1))&(BB(:,1)<lim(2)))); % bounding box size and location
    

[N1,edges1] = histcounts(Area);
[~,mx1] = max(N1);
try
    lim1 = [edges1(mx1-1),edges1(mx1+1)];
catch
    lim1 = [edges1(mx1),edges1(mx1+2)];
end
fx3 = find(Area>lim1(1) & Area<lim1(2)); % Area

fx = intersect(fx2,fx3);

nBB = BB(fx,:);
h = levcook(nBB(:,1),nBB(:,2));
out_idx = find(h>2.5*std(h));

fx(out_idx) = [];
nBB = BB(fx,:);

reg_props1 = reg_props(fx);

if  show == 1
    figure(20);
    imshow(RGB1); hold on;
end

BBw = false(size(BW));
for ii = 1:length(fx)
    if show == 1
        rectangle('Position',nBB(ii,:),'EdgeColor','g','LineWidth',2)
        hold on;
    end
    BBw(reg_props1(ii).PixelIdxList) = true;
end

sq_dims = median(nBB(:,3:4));

BW = true(sz(1:2));
BW(:,(round(sz(2)/2):end),:) = not(imdilate(BBw,strel('disk',3)));

try
    BW1 = bwconvhull(not(BW));
    reg_fin = regionprops(BW1,'BoundingBox');
    bb1 = reg_fin.BoundingBox;
    
    if ((sz(1)/sz(2))>1.5) && Rside(1)==1 % long right
        bb2 = [0.90*bb1(1), 0, 1.05*bb1(3),0.40*sz(1)]; %rhs
        bb3 = [bb1(1)-0.6*sz(2), 0, 1.25*bb1(3), sz(1)]; %lhs
    elseif ((sz(1)/sz(2))>1.5) && Rside(2)==1 % long left
        bb2 = [0.99*bb1(1), 0, 0.5*bb1(3),sz(1)];
        bb3 = [bb1(1)-0.55*sz(2), 0, 1.15*bb1(3), 0.50*sz(1)]; %lhs;
    else
        bb2 = [0.90*bb1(1), 0, 1.05*bb1(3),sz(1)];
        bb3 = [bb1(1)-0.5*sz(2), 0, 1.025*bb1(3), sz(1)];
    end

    pts = round(bbox2points(bb2));
    pts(2:3,1) = [sz(2);sz(2)];
    BW2 = poly2mask(pts(:,1),pts(:,2),sz(1),sz(2));
    
    pts1 = round(bbox2points(bb3));
    pts1(2:3,1) = [0;0];
    BW21 = poly2mask(pts1(:,1),pts1(:,2),sz(1),sz(2));
    
    BW3 = or(BW2,BW21);

    BW = not(or(BW3,BW1));
end
% Initialize output masked image based on input image.
maskedRGBImage = RGB;

% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

end
