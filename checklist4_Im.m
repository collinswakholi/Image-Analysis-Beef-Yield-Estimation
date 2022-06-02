function [exists,Pos] = checklist4_Im(list_array,Im_name,ccs)

len = length(list_array);

Name = {};
for i=1:len
    list1 = list_array{1,i};
    Name2 = {};
    for ii = 1:length(list1)
        name2 = extract_name(list1{ii,1},ccs);
        Name2 = [Name2;name2];
    end
    Name{i} = Name2;
end

Xst = [];
Pos = [];
for i = 1:len
    [pos,xst] = find(contains(Name{1,i},Im_name));
    Xst = [Xst,xst];
    Pos = [Pos,pos];
end
exists = sum(Xst)==len;