function [I_points,coord] = Im_points(I, n, pt_sz)

if ~exist('pt_sz')
    pt_sz = 3;
end

if ~exist('n')
    n = 11;
end

aaa = imshow(I);

zz = n;
coord = zeros(n,3);

while zz > 0.5
    if ishandle(aaa)
        try
            hold on;
            [x,y] = ginput(1);
            scatter(x,y,100,'o','LineWidth',0.6,'MarkerEdgeColor', 'k','MarkerFaceColor','g')
            txt = input('enter number\n','s');
            text(x+5,y+5,txt,'FontSize',20)
            coord(str2num(txt),:) = [1, x, y];
            zz = zz - 1
        end
    else
        zz = 0;
    end
end

I_points = zeros(size(I,1), size(I,2));
coord2 = coord((sum(coord,2)~=0),:);
for i = 1:size(coord2,1)
    yx = round(coord2(i,:));
    I_points(yx(3),yx(2)) = 1;
end

I_points = imdilate((I_points>0), strel('disk',pt_sz));