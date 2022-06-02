% read images

function Imgs  = readImages(imNames_1,imNames_9,imNames_3,imNames_4,ccs)

Im_934 = {imNames_9,imNames_3,imNames_4};

[Imgs_1,Imgs_9,Imgs_3,Imgs_4] = deal([]);
Name = {};
k=0;
for i = 1:length(imNames_1)
    name = extract_name(imNames_1{i,1},ccs);
    [go,pos] = checklist4_Im(Im_934,name,ccs);
    I_1 = imread(imNames_1{i,1});

    if go == 1
        I_9 = imread(imNames_9{pos(1),1});
        I_3 = imread(imNames_3{pos(2),1});
        I_4 = imread(imNames_4{pos(3),1});
        k=k+1;
    else
%         [I_1, I_9, I_3, I_4] = deal([]);
        name = [];
    end
    Name = [Name;name]; 
    Imgs_1{k,1} = I_1;
    Imgs_9{k,1} = I_9;
    Imgs_3{k,1} = I_3;
    Imgs_4{k,1} = I_4;
    disp([num2str(i),'/',num2str(length(imNames_1))])
end

Imgs{1} = Imgs_1;
Imgs{2} = Imgs_9;
Imgs{3} = Imgs_3;
Imgs{4} = Imgs_4;