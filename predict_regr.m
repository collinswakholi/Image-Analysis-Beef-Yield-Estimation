function Iz = predict_regr(I,mdl,r)

% I input image
% r resize ratio for resizing the image
% I_pred foreground prediction from image
% I = Ip;

if exist('r','var')
    I = imresize(I,r);
end

[~,IIp] = find_remove_checkers(I);
[~,IIIp] = remove_bs(IIp);
sz1 = size(IIIp);

    HSVp = rgb2hsv(IIIp);
    XYZp = rgb2xyz(IIIp);
    LABp = rgb2lab(IIIp);
    ycbcrp = rgb2ycbcr(IIIp);
    AA = mat2gray(LABp(:,:,2));
    RA = (double(IIIp(:,:,1)).*(2.5*AA));

    I_allp = double(cat(3,IIIp,100*HSVp,100*XYZp,LABp,ycbcrp));

Ipr = reshape(I_allp, sz1(1)*sz1(2),size(I_allp,3));

Ip_pred = predict(mdl,Ipr);

I_pred = reshape(Ip_pred,sz1(1),sz1(2));

I_pred(I_pred>1) = 1;
I_pred(I_pred<0) = 0;

I_pred = uint8(255*I_pred);

Iz = cat(3,double(I_pred),RA);