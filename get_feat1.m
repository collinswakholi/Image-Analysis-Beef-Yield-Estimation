function feat = get_feat1(Im,mask,ccs,keypoints,dl_mask)

if ccs.deepLearn == 0
    dl_feat = [];
    do_dl = 0;
else
    do_dl =1;
end

% resize_factor = 1;
plt = ccs.plt; % 1 is for plot results, 0 for no plot
% sve = 0; % 0 for dont save results, 1 for save results

% mask standard frames load
if strcmp(ccs.side, 'Right')&& strcmp(ccs.state, 'Warm')
    I_ref = keypoints.WR{1, 1};
    mask_out = keypoints.WR{1, 2};
    mask_in = keypoints.WR{1, 3};
    Rig = 1;
    
elseif strcmp(ccs.side, 'Right')&& strcmp(ccs.state, 'Cold')
    I_ref = keypoints.CR{1, 1};
    mask_out = keypoints.CR{1, 2};
    mask_in = keypoints.CR{1, 3};
    Rig = 1;
    
elseif strcmp(ccs.side, 'Left')&& strcmp(ccs.state, 'Warm')
    I_ref = keypoints.WL{1, 1};
    mask_out = keypoints.WL{1, 2};
    mask_in = keypoints.WL{1, 3};
    Rig = 0;
    
elseif strcmp(ccs.side, 'Left')&& strcmp(ccs.state, 'Cold')
    I_ref = keypoints.CL{1, 1};
    mask_out = keypoints.CL{1, 2};
    mask_in = keypoints.CL{1, 3};
    Rig = 0;
end


%% mask region props
properties = regionprops(mask,...
        'Area',...
        'BoundingBox',...
        'Centroid',...
        'Extent',...%Ratio of pixels in the region to pixels in the total bounding box,
        'MajorAxisLength',...
        'MinorAxisLength',...
        'Perimeter',...
        'Orientation');
    
% Extract boundary coordinates
outline = bwboundaries(mask);
[~, max_index] = max(cellfun('size', outline, 1));
outline = outline{max_index,1};

     %% plot
     if plt ==1
     figure()
        imshow(Im)
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
%% FACTORS (A, B, C, D) image processing
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
        scatter(X0,Y0,200,'+','m','LineWidth',1.5);
        hold on;
        line(X00,Y00,'Color','blue','LineStyle','-.','LineWidth',1.5);
        pause(.01)
        end
        
        coord_X0 = [coord_X0;X0];
        coord_Y0 = [coord_Y0;Y0];
    end
    
    cood_XY0 = [coord_X0,coord_Y0]; % intersect of centroid-box plot corners line with boundary
    
    diff = centroid-cood_XY0;
    A_fx = hypot(diff(:,1),diff(:,2));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% A_fx
    A_coord = [cood_XY0,repmat(centroid,4,1)];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% A_fx
    
%     horizontal nth percentile of length
    B_fx = [];
    C_fx = [];
    C_points = [];
    X11 = [];
    Y11 = [];
    B_coord = [];
    C_coord =[];
    
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
            pause(.01)
%         
        end
        
        % distace between end to end of outline
        diff_perc = [(X1(1)-X1(2)),(Y1(1)-Y1(2))];
        dist_perc = hypot(diff_perc(1),diff_perc(2));
        B_fx = [B_fx, dist_perc];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% B_fx
        B_coord = [B_coord;[X1(1),Y1(1),X1(2),Y1(2)]]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%B_coord
        
        
        % distance between outline B-coord coordinates and centroid
        diff_perc1 = [(X1-centroid(1)),(Y1-centroid(2))];
        dist_perc1 = hypot(diff_perc1(:,1),diff_perc1(:,2));
        C_fx = [C_fx, dist_perc1'];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% C_fx
        C_coord = [C_coord;[X1,Y1,repmat(centroid,2,1)]]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%C_coord
    end
    
    %     distances between boundary points
    D_fx = [];
    D_coord = [];
    lenz = length(X11);
    for ii = 1:lenz
        diff_out_coord = [(X11-X11(ii)),(Y11-Y11(ii))];
        dist_out_coord = hypot(diff_out_coord(:,1),diff_out_coord(:,2));
        D_fx = [D_fx,dist_out_coord'];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D_fx
        D_coord = [D_coord;[X11,Y11,repmat([X11(ii),Y11(ii)],lenz,1)]];%%%%%%%%%%%%%% D_coord
    end
    if plt
        figure;
        
        plot_points_lines(mask,D_coord,'r','b',1.5,0.75);hold on;
        plot(outline(:,2),outline(:,1),...
                'g',...
                'LineWidth',2); hold on;
    end
    
    for hh=1:length(B_fx)
        B_name{hh} = strcat('B',num2str(hh));
    end

    for m=1:length(C_fx)
        C_name{m} = strcat('C',num2str(m));
    end

    for n=1:(length(D_fx))
        D_name{n} = strcat('D',num2str(n));
    end
    
%%
%% FACTORS II (E) fitting standard carcas feature points on new carcass
%%
    Coords = match_keypoints(I_ref,Im,mask_out,mask_in,mask,plt,1);
    coord_refined_orig = Coords.output.cc;
    coord_order = Coords.output.idx;
    coord_refined = coord_refined_orig(coord_order,:);
    
    %% find distances between points
    X_refined = double(coord_refined(:,1));
    Y_refined = double(coord_refined(:,2));

        E_fx = [];
        E_coord = [];
        lens = length(coord_refined);
    for iii = 1:lens
        diff_coord_refined = [(X_refined-X_refined(iii)),(Y_refined-Y_refined(iii))];
        dist_coord_refined = hypot(diff_coord_refined(:,1),diff_coord_refined(:,2));
        E_fx = [E_fx,dist_coord_refined'];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% E_fx (Fx2)
        E_coord = [E_coord;[coord_refined,repmat(coord_refined(iii,:),lens,1)]];
    end
    if plt
        figure;
%         plot_points(Im,E_coord(:,1:2),'b')
        plot_points_lines(Im,E_coord,'r','b',1.8,1);hold on;
        plot(outline(:,2),outline(:,1),...
                'g',...
                'LineWidth',2); hold on;
    end
    
    
    for nn=1:(length(E_fx))
        E_name{nn} = strcat('E',num2str(nn));
    end
    

 %%
%% DEEP LEARNING/ depth based contour features
%%
if do_dl == 1
    % keep dl prediction between O10 and O12
    idx10 = find(coord_order == 10);
    idx12 = find(coord_order == 12);
    coord_O10 = coord_refined_orig(idx10,:);
    coord_O12 = coord_refined_orig(idx12,:);
    dl_mask1 = dl_mask;
    dl_mask1(1:coord_O10(2),:) = 0;
    dl_mask1(coord_O12(2):end,:) = 0;
    
    dl_exist = find(dl_mask1==1);
    if length(dl_exist)>10
        dl_mask = dl_mask1;
    end
    
    outline_dl = bwboundaries(dl_mask);
        
        [~, max_index1] = max(cellfun('size', outline_dl, 1));
        outline_dl = outline_dl{max_index1,1};
        outline_dl = [outline_dl(:,2),outline_dl(:,1)];
        len_dl = length(outline_dl);
        
        %find closest point to centroid
        diff_dl = outline_dl-centroid;
        dist_dl = hypot(diff_dl(:,1),diff_dl(:,2));
        [F1_Fx,pos] = min(dist_dl);
        coord_dl_cent = outline_dl(pos,:);
        F1_Fx_coord = [coord_dl_cent,centroid];

        
        %find distances from contour to closest defined points
        F2_Fx = [];
        dl_detPts = [];
        for ii = 1:length(coord_refined)
            coord1 = double(coord_refined(ii,:));
            diff_dl1 = outline_dl-coord1;
            dist_dl1 = hypot(diff_dl1(:,1),diff_dl1(:,2));
            [g_Fx,pos] = min(dist_dl1);
            coord_dl_detPts = outline_dl(pos,:);
            F2_Fx = [F2_Fx,g_Fx];
            dl_detPts = [dl_detPts;coord_dl_detPts];
        end
        F2_Fx_coord = [dl_detPts,coord_refined];

        
        %find closest points to 32 outline points
        F3_Fx = [];
        dl_outline = [];
        for ii = 1:length(C_points)
            coord2 = C_points(ii,:);
            diff_dl2 = outline_dl - coord2;
            dist_dl2 = hypot(diff_dl2(:,1),diff_dl2(:,2));
            [h_Fx,pos] = min(dist_dl2);
            coord_dl_outline = outline_dl(pos,:);
            F3_Fx = [F3_Fx,h_Fx];
            dl_outline = [dl_outline;coord_dl_outline];
        end
        F3_Fx_coord = [dl_outline,C_points];
        
        F_Fx = [F1_Fx,F2_Fx,F3_Fx];
        F_coord = [F1_Fx_coord;F2_Fx_coord;F3_Fx_coord];
        
        for nn=1:(length(F_Fx))
            F_name{nn} = strcat('F',num2str(nn));
        end
        % plot all
        if plt ==1
            Im1 = 2*imfuse(Im,bwmorph(dl_mask,'fatten',3),'blend');
            plot_points_lines(Im1, F_coord,'g','b',1.8,1)
        end
else
    F_Fx = [];
    F_name = {};
    F_coord = [];
end


%% compile features and respective names

fx_names = {'area', 'BB_width', 'BB_length', 'perimeter', 'major', 'minor',...
        'extent','orientaion','A1','A2','A3','A4',B_name{:},C_name{:},D_name{:},E_name{:},F_name{:}};
    
fx_values = [properties.Area,BB_box(3:4),properties.Perimeter,...
        properties.MajorAxisLength,properties.MinorAxisLength,...
        properties.Extent,properties.Orientation,A_fx',B_fx,...
        C_fx,D_fx,E_fx,F_Fx];
    
fx_coord.RP = properties; %Rp properties
fx_coord.A = A_coord; %A_features
fx_coord.B = B_coord; %B_features
fx_coord.D = D_coord; %C_features
fx_coord.C = C_coord; %D_features
fx_coord.E = E_coord; %E_features
fx_coord.F = F_coord; %F_features
    
feat = {fx_names,fx_values,fx_coord};
