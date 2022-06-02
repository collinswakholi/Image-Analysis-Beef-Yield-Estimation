function plot_polygons(I,polys)



imshow(I); hold on;
for i = 1:length(polys)
    plot(polys{1,i}); hold on;
end
hold off;
drawnow