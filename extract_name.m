function name = extract_name(dir_str,ccs)


in_str = strsplit(dir_str,'\');
str2 = strsplit(in_str{end},'_');

if strcmp(ccs.state, 'Cold')
    str2{1,2} = num2str((str2num(str2{1,2})-1),'%.4d');
end
str3 = strsplit(str2{3},' ');

name = strjoin({str2{1},str2{2},str3{end}},'_');

