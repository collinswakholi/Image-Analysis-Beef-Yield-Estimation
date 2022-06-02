function pts = get_points(I)
% get point from images
aaa = imshow(I,[]);

pts = [];
a = 1;

while a>0
    if ishandle(aaa)
        try
            hold on;
            [x,y] = ginput(1);
            scatter(x,y,200,'*','g','LineWidth',2);
        end
    else
        a = 0;
    end
    pts = round([pts;[x,y]]);
end