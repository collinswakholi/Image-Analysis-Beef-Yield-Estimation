% Labels selected points on an image and returns their coordinates and
% given indices
% Collins(wcoln@yahoo,com) Jan 2020

function pointLabels = LabelImagePoints(I)
close
imshow(I);
pointLabels = [];
nn = 1;
while nn > 0
    hold on;
    [x,y] = ginput(1);
    scatter(x,y,100,'o','LineWidth',0.6,'MarkerEdgeColor', 'k','MarkerFaceColor','c')
    idx = inputdlg('Please enter point index!!! (Enter zero when done selecting)');
    idx = str2double(idx{1,1});
    if idx~=0
        pointLabels = [pointLabels;[idx,x,y]];
        text(x-5,y-20,num2str(idx),'FontSize',14,'Color','green','HorizontalAlignment','center')
        title(['Total number of points = ',num2str(nn)])
        drawnow
        nn=nn+1;
    else
        nn=0;
    end
end

