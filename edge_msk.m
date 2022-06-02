function BW =  edge_msk(RGB)

Ji = rangefilt(RGB);
LAB = rgb2lab(RGB);
% II_s = LAB(:,:,2).*mat2gray(Ji(:,:,1));
% BW = imbinarize(II_s,graythresh(II_s));

edg = edge(LAB(:,:,2),'prewitt');
BW = imclose(edg,strel('disk',5));
BW = imfill(BW,'holes');
BW = bwareafilt(BW,1);
