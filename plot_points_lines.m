function plot_points_lines(I,coord,c1,c2,wid1,wid2,ln_sytle)

% coor must be an nx4 matrix
if ~exist('c1','var')
    c1 = 'r';
end
if ~exist('c2','var')
    c2 = 'b';
end

if ~exist('wid1','var')
    wid1 = 1.5;
end

if ~exist('wid2','var')
    wid2 = 1.5;
end

if ~exist('ln_sytle','var')
    ln_sytle = '-';
end

imshow(I); hold on;
for i = 1:size(coord,1)
    line([coord(i,1);coord(i,3)],...
                [coord(i,2);coord(i,4)],...
                'Color',c2,'LineStyle',ln_sytle,'LineWidth',wid2); hold on;
end

hold on;
scatter(coord(:,1),coord(:,2),150,'*',c1,'LineWidth',wid1); hold on;
scatter(coord(:,3),coord(:,4),150,'*',c1,'LineWidth',wid1); hold on;

drawnow