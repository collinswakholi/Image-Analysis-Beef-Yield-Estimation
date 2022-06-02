clear; clc;

pt_dir = uigetdir;
files = dir(pt_dir);
files(1:2,:) = [];

% points
Pts = [];
for i = 1:length(files)
    x1 = load([pt_dir, '\' files(i).name]);
    pts = round(x1.pts);
    
    if size(pts,1) ~= 11
        pts = pts(end-10:end,:);
%         i
    end
    
    ptr = [];
    for j = 1:11
        ptr = [ptr, pts(j,2:end)];
    end
    
    ptsr = [pts(:,1)', ptr];
    Pts = [Pts;ptsr];
end
% save('Pts.mat','Pts');
% xlswrite('Pts.xlsx',Pts);
writematrix(Pts,'Pts.csv')