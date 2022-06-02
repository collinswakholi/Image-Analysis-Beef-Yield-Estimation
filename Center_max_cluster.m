function [CC,pos] = Center_max_cluster(C,coord)

mini = min(C);
maxi = max(C);

Med = [];
for i = mini:maxi
    cordin = coord(find(C==i),:);
    if length(cordin)>2
        med = median(cordin);
    else
        med = [];
    end
%     scatter(cordin(:,1),cordin(:,2)); hold on;scatter(med(1),med(2),100,'r')
    Med = [Med;med];
end

[~,best] = max(hypot(Med(:,1),Med(:,2)));

CC = Med(best,:);

pos = knnsearch(coord,CC);