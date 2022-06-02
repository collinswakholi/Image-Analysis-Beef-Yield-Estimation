function [BW, sq_dims] = get_mask_improc(Im, mdl)

se = strel('disk',7);
se1 = strel('disk',3);
se2 = strel('disk',2);

sz = size(Im);
[~,II] = bb_2(Im);
try
[~,III,sq_dims] = find_checkers(II,ss);
catch
    III = II;
    sq_dims = [31,31];
end
III_r = imresize(III,0.3);
I_pr = my_predict_regress(III_r,mdl,intx);
I_pr(I_pr<0) = 0;
I_pr(I_pr>1) = 1;

Bbz = Bim_segpca(I_pr);
Bbz = activecontour(I_pr,Bbz,10);
edg = imclose(edge(I_pr,'prewitt'),se1);

Mask = get_center_blob(Bbz,edg);
Bbw = bwareafilt((imopen(Mask,se2)),1);

BW = medfilt2(imresize(Bbw, sz(1:2),'bicubic'),[3,3]);