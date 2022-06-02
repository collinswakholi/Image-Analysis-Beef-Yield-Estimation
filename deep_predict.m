function [Pred,I_new] = deep_predict(I,netSz,brk)

if nargin<3
    brk = 0;
end

    loc = 'F:\Collins_ops\Carcass_data_analysis_2020\deep_learning_model';
%     delete([loc,'/predicted.png'])
if brk == 0
    % make image square
    Im_sq = {make_square_img(I,netSz)};
else
    % split and make square
    Im_sq = sq_split_im(I,netSz);
end

BB = {};
for i1 = 1:length(Im_sq)
    if isfile([loc,'\predicted.png'])
        delete([loc,'\predicted.png'])
        drawnow
    end
    
    % write image
    imwrite(Im_sq{1,i1}, [loc,'\Im_1.png'])
   
    pause(0.3)
    drawnow
    % read image if available
    i =1;
    while i > 0
        try
            I_new = imread([loc,'\predicted.png']);

    %         edg = edge(rgb2gray(I_new),'log');
    %         BW = imfill(edg,'holes');
    %         BW = bwmorph(BW,'thin',inf);
            i = 0;
        catch 
                clc; disp(['waiting for network_(', num2str(i),'_iterations)'])
                drawnow
                i = i+1;
        end
    end
    waitfor(I_new)
    BB{i1,1} = I_new;
end 

if brk == 0
    BW = make_orig_size(BB{1},size(I));
else
    BW = make_orig_from_halves(BB,size(I));
end

Pred = BW;