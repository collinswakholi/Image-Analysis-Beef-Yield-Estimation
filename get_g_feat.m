function [G_feat, G_names, G_poly] = get_g_feat(coord_order,polygons)
% function extract area from the specified polygons using keypoints
triangles = polygons.Triangles;
quads = polygons.Quads;


Tr_name = {};
Qd_name = {};
Area_tr = [];
Area_qd = [];
Tr_polygon = {};
Qd_polygon = {};

for ix = 1:30
    tr = triangles(ix,:);
    tr_name = strjoin({num2str(tr(1)),num2str(tr(2)),num2str(tr(3))},'-');
    Tr_name{ix} = tr_name;
    coords_tr = coord_order(tr,:);
    
    tr_polygon = polyshape(coords_tr(:,1),coords_tr(:,2));
    
    area_tr = area(tr_polygon);
    Area_tr = [Area_tr, area_tr];
    
    Tr_polygon{ix} = tr_polygon;
end

for ix = 1:20
    qd = quads(ix,:);
    qd_name = strjoin({num2str(qd(1)),num2str(qd(2)),num2str(qd(3)),num2str(qd(4))},'-');
    Qd_name{ix} = qd_name;
    coords_quad = coord_order(qd,:);
    
    qd_polygon = polyshape(coords_quad(:,1), coords_quad(:,2));
    
    area_qd = area(qd_polygon);
    Area_qd = [Area_qd, area_qd];
    
    Qd_polygon{ix} = qd_polygon;
end

% names
g_names_tr = strcat('G(',Tr_name,')');
g_names_qd = strcat('G(',Qd_name,')');

G_names = [g_names_tr,g_names_qd];


% values
G_feat = [Area_tr, Area_qd];

% polygons
G_poly = [Tr_polygon,Qd_polygon];

