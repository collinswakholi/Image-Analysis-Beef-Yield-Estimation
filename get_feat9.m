% clear all, close all, clc;

function feat = get_feat9(Im,mask,ccs)

plt = ccs.plt; % 1 is for plot results, 0 for no plot

%% mask region props
Img = Im;
msk =  mask;

properties = regionprops(msk,...
        'Area',...
        'BoundingBox',...
        'Centroid',...
        'Extent',...%Ratio of pixels in the region to pixels in the total bounding box,
        'MajorAxisLength',...
        'MinorAxisLength',...
        'Perimeter',...
        'Orientation');
    
% Extract boundary coordinates
outline = bwboundaries(msk);
[~, max_index] = max(cellfun('size', outline, 1));
outline = outline{max_index,1};

     %% plot
     if plt ==1
     figure()
        imshow(Img)
        hold on;
        rectangle('Position',properties.BoundingBox,...
            'EdgeColor', 'r',...
            'LineWidth',2,'LineStyle','-')
        hold on;
        scatter(properties.Centroid(1),properties.Centroid(2),200,'*',...
            'MarkerEdgeColor',[1 0 1],...
            'MarkerFaceColor',[.7 .7 .7],...
            'LineWidth',2);
        hold on;
            plot(outline(:,2),outline(:,1),...
            'g',...
            'LineWidth',2);
     end
     
%%     
%% FACTORS (A, B, C) image processing
%%
    %line centroid_boxPlot corners intersection with carcass outline
    BB_box = properties.BoundingBox;
    BB_coord = bbox2points(BB_box);
    centroid = properties.Centroid;
    
    coord_X0 = [];
    coord_Y0 = [];
    
    for i = 1:4
        [X0 Y0] = polyxpoly([BB_coord(i,1);centroid(1)],...
            [BB_coord(i,2);centroid(2)],...
            outline(:,2),outline(:,1));% find intersection
        X0 = mean(X0);
        Y0 = mean(Y0);
        X00 = [X0;centroid(1)];
        Y00 = [Y0;centroid(2)];
        
        if plt == 1
            
        hold on;
        scatter(X0,Y0,200,'*','R','LineWidth',2);
        hold on;
        line(X00,Y00,'Color','blue','LineStyle','-.','LineWidth',2);
        pause(.01)
        end
        
        coord_X0 = [coord_X0;X0];
        coord_Y0 = [coord_Y0;Y0];
    end
    
    cood_XY0 = [coord_X0,coord_Y0]; % intersect of centroid-box plot corners line with boundary
    
    diff = centroid-cood_XY0;
    A_fx = hypot(diff(:,1),diff(:,2));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% A_fx
    A_coord = [cood_XY0,repmat(centroid,4,1)];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% A_coord
    
%     horizontal nth percentile of length
    B_fx = [];
    C_fx = [];
    B_coord = [];
    C_coord = [];
    C_points = [];
    X11 = [];
    Y11 = [];
    
     if plt
        figure(2);
        imshow(Im);hold on;
        plot(outline(:,2),outline(:,1),...
                'g',...
                'LineWidth',2); hold on;

        scatter(properties.Centroid(1),properties.Centroid(2),200,'*',...
                'MarkerEdgeColor',[1 0 1],...
                'MarkerFaceColor',[.7 .7 .7],...
                'LineWidth',2); hold on;
     end
    
    for i = 1:16 % depends on number of segments you want to make (carcass divided into 16 along the body)
        percentile = i/20;
        BB_box_10 = BB_box;
        BB_box_10(4) = percentile*BB_box(4);
        BB_coord_10 = bbox2points(BB_box_10);

%         find intersection between BB_coord and outline
        [X1 Y1] = polyxpoly(BB_coord_10(3:4,1),...
            BB_coord_10(3:4,2),...
            outline(:,2),outline(:,1));%find intesection
        X1 = [X1(1);X1(end)];
        Y1 = [Y1(1);Y1(end)];
        X11 = [X11;(X1)];
        Y11 = [Y11;(Y1)];
        
        C_pts = [X1,Y1];
        C_points = [C_points;C_pts];

        if plt ==1
            
            scatter(X1,Y1,100,'*','r','LineWidth',1.5);
            hold on;
            line(X1,Y1,'Color','b','LineStyle','-.','LineWidth',1.5);
            hold on;
            drawnow

    %         hold on;
            line([X1(1);centroid(1)],...
                [Y1(1);centroid(2)],...
                'Color','b','LineStyle','--','LineWidth',1.5);
            hold on;
            line([X1(2);centroid(1)],...
                [Y1(2);centroid(2)],...
                'Color','b','LineStyle','--','LineWidth',1.5);
        
        end
        
        % distace between end to end of outline
        diff_perc = [(X1(1)-X1(2)),(Y1(1)-Y1(2))];
        dist_perc = hypot(diff_perc(1),diff_perc(2));
        B_fx = [B_fx, dist_perc];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B_fx
        B_coord = [B_coord;[X1(1),Y1(1),X1(2),Y1(2)]]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%B_coord
        
        % distance between outline coordinates and centroid
        diff_perc1 = [(X1-centroid(1)),(Y1-centroid(2))];
        dist_perc1 = hypot(diff_perc1(:,1),diff_perc1(:,2));
        C_fx = [C_fx, dist_perc1'];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% C_fx
        C_coord = [C_coord;[X1,Y1,repmat(centroid,2,1)]]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C_coord
    end
    
    %     distances between boundary points
%     D_fx = [];
%     for ii = 1:length(X11)
%         diff_out_coord = [(X11-X11(ii)),(Y11-Y11(ii))];
%         dist_out_coord = hypot(diff_out_coord(:,1),diff_out_coord(:,2));
%         D_fx = [D_fx,dist_out_coord'];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D_fx
%     end

    for hh=1:length(B_fx)
        B_name{hh} = strcat('B',num2str(hh));
    end

    for m=1:length(C_fx)
        C_name{m} = strcat('C',num2str(m));
    end

%     for n=1:(length(D_fx))
%         D_name{n} = strcat('D',num2str(n));
%     end
    


%% compile features and respective names

fx_names = {'area', 'BB_width', 'BB_length', 'perimeter', 'major', 'minor',...
        'extent','orientaion','A1','A2','A3','A4',B_name{:},C_name{:}};
    
fx_values = [properties.Area,BB_box(3:4),properties.Perimeter,...
        properties.MajorAxisLength,properties.MinorAxisLength,...
        properties.Extent,properties.Orientation,A_fx',B_fx,...
        C_fx];
    
    fx_coord.RP = properties; %Rp properties
fx_coord.A = A_coord; %A_features
fx_coord.B = B_coord; %B_features
fx_coord.C = C_coord; %C_features
    
feat = {fx_names,fx_values,fx_coord};
