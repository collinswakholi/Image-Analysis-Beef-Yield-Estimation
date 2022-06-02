imds = imageDatastore('F:\Collins_ops\Deep_learning\Deep_201906\PixelLabelData');
for i=1:length(imds.Files)
    name = imds.Files{i,1};
    split = strsplit(name,'\');
    split2 = strsplit(split{1,6},'_');
    split3 = strsplit(split2{1,2},'.');
    number = str2num(split3{1,1});
    saveName = strcat(strjoin(split(1:5),'\'),'\',split2{1,1},'_',num2str(i,'%4.4d'),'.png');
    I = imread(name);
    imwrite(I,saveName)
end