function IDX = chooseFeat2(feat_names,images,feat4model)

[idx1,idx1A,idx1B,idx1C,idx1D,idx1E,idx1F,idx1G...
    idx9,idx9A,idx9B,idx9C,...
    idx3,idx3A,idx3B,idx3C,...
    idx4,idx4A,idx4B,idx4C] = deal([]);

if sum(images==1) == 1 % for image 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx1 = find(contains(feat_names,'_1'));
    if sum(contains(feat4model,'A')) == 1
        idx1A = [1:find(contains(feat_names,'A4_1'))];% 12 feat
    end
    if sum(contains(feat4model,'B')) == 1
        idx1B = [find(contains(feat_names,'B1_1')):find(contains(feat_names,'B16_1'))]; % 16 feat
    end
    if sum(contains(feat4model,'C')) == 1
        idx1C = [find(contains(feat_names,'C1_1')):find(contains(feat_names,'C32_1'))]; % 32 feat
    end
    if sum(contains(feat4model,'D')) == 1
        idx1D = [find(contains(feat_names,'D1_1')):find(contains(feat_names,'D1024_1'))]; % 1024 feat
    end
    if sum(contains(feat4model,'E')) == 1
        idx1E = [find(contains(feat_names,'E1_1')):find(contains(feat_names,'E256_1'))]; % 484 feat
    end
    if sum(contains(feat4model,'F')) == 1
        idx1F = [find(contains(feat_names,'F1_1')):find(contains(feat_names,'F49_1'))]; % 55 feat
    end
    if sum(contains(feat4model,'G')) == 1
        idx1G = find(contains(feat_names,'G('));                                        % 50 feat
    end
end

if sum(images==9) == 1 % for image 9 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx9 = find(contains(feat_names,'_9'));
    if sum(contains(feat4model,'A')) == 1
        idx9A = [find(contains(feat_names,'area_9')):find(contains(feat_names,'A4_9'))];% 12 feat
    end
    if sum(contains(feat4model,'B')) == 1
        idx9B = [find(contains(feat_names,'B1_9')):find(contains(feat_names,'B16_9'))]; % 16 feat
    end
    if sum(contains(feat4model,'C')) == 1
        idx9C = [find(contains(feat_names,'C1_9')):find(contains(feat_names,'C32_9'))]; % 32 feat
    end
end

if sum(images==3) == 1 % for image 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx3 = find(contains(feat_names,'_3'));
    if sum(contains(feat4model,'A')) == 1
        idx3A = [find(contains(feat_names,'area_3')):find(contains(feat_names,'A4_3'))];% 12 feat
    end
    if sum(contains(feat4model,'B')) == 1
        idx3B = [find(contains(feat_names,'B1_3')):find(contains(feat_names,'B16_3'))]; % 16 feat
    end
    if sum(contains(feat4model,'C')) == 1
        idx3C = [find(contains(feat_names,'C1_3')):find(contains(feat_names,'C32_3'))]; % 32 feat
    end
end

if sum(images==4) == 1 % for image 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx4 = find(contains(feat_names,'_4'));
    if sum(contains(feat4model,'A')) == 1
        idx4A = [find(contains(feat_names,'area_4')):find(contains(feat_names,'A4_4'))];% 12 feat
    end
    if sum(contains(feat4model,'B')) == 1
        idx4B = [find(contains(feat_names,'B1_4')):find(contains(feat_names,'B16_4'))]; % 16 feat
    end
    if sum(contains(feat4model,'C')) == 1
        idx4C = [find(contains(feat_names,'C1_4')):find(contains(feat_names,'C32_4'))]; % 32 feat
    end
end

IDX = [idx1A,idx1B,idx1C,idx1D,idx1E,idx1F,idx1G,idx9A,idx9B,idx9C,idx3A,idx3B,idx3C,idx4A,idx4B,idx4C];