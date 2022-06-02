function Mask = get_center_blob(mask,edg) % based on mask & edge

sz = size(mask);
mx = round(mean(sz)/30);

center = (round(sz/2));
center = [center(2),center(1)];
% mask
reg = regionprops(mask,'PixelList','PixelIdxList');

Md = [];
for iz = 1:length(reg)
    pix = reg(iz).PixelList;
%     [~,dist] = knnsearch(center,pix);
    [~,min_dist] = knnsearch(pix,center);
    Md = [Md,min_dist];
end

min_Dist = min(Md);

Ptz = {};
for iz = 1:length(reg)
    pix = reg(iz).PixelList;
    [~,dist] = knnsearch(center,pix);
    Idz = find(dist<=20*min_Dist);
    ptz = pix(Idz);
    Ptz{iz} = length(ptz)/size(pix,1);
end

[~,Idx] = max(cellfun(@max,Ptz));

Msk = zeros(sz);
Msk(reg(Idx).PixelIdxList) = 1;
reg = regionprops(Msk,'PixelList','PixelIdxList');
coord = reg.PixelList;

% edge
reg2 = regionprops(edg,'PixelList','PixelIdxList');

Ptz = {};
for ij = 1:length(reg2)
    pixe = reg2(ij).PixelList;
    [~,dist2] = knnsearch(coord,pixe);
    Idv = find(dist2<mx);
    ptz = pixe(Idv);
    Ptz{ij} = length(ptz)/size(pixe,1);
end
Ptz = cell2mat(Ptz);
Idx = find(Ptz>0.5);

Msk2 = zeros(sz);
for ik = 1:length(Idx)
    k = Idx(ik);
    Msk2(reg2(k).PixelIdxList) = 1;
end

Mask = or(Msk,Msk2);