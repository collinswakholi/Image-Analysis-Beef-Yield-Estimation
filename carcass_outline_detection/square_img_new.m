function [image,mask] = square_img_new(I,msk,size1,half)
% gets an image, converts it to square image of dimensions size1
% also randomely crops images with aspect ratio more than 1.4:1 or visevasa

rn = rand;
% image = uint8(zeros(size1,size1));

sz = size(I);
[~, i] = max(sz);
ext = round(0.5*(abs(sz(1) - sz(2))));

ar = sz(i)/(sz(i)-2*ext);
if (rn>=0.7) && (ar>1.4) && (half == 1)
    if i==1
        imT = I(1:round(end/2),:,:);
        imB = I(round(end/2)+1:end,:,:);
        
        msT = msk(1:round(end/2),:);
        msB = msk(round(end/2)+1:end,:);
    else
        imT = I(:,1:round(end/2),:);
        imB = I(:,round(end/2)+1:end,:);
        
        msT = I(:,1:round(end/2));
        msB = I(:,round(end/2)+1:end);
    end
    ran = (rand)>0.5;
    if ran == 1
        im1 = imT;
        ms1 = msT;
    else
        im1 = imB;
        ms1 = msB;
    end
else
    if i == 1
        im1(:,:,1) = padarray(I(:,:,1),[0 ext],'both');
        im1(:,:,2) = padarray(I(:,:,2),[0 ext],'both');
        im1(:,:,3) = padarray(I(:,:,3),[0 ext],'both');
        
        ms1 = padarray(msk,[0 ext],'both');
    else
        im1(:,:,1) = padarray(I(:,:,1),[ext 0],'both');
        im1(:,:,2) = padarray(I(:,:,2),[ext 0],'both');
        im1(:,:,3) = padarray(I(:,:,3),[ext 0],'both');
        
        ms1 = padarray(msk,[ext 0],'both');
    end
end



image = imresize(im1,[size1 size1]);
mask = imresize(ms1,[size1 size1]);