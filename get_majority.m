function cutoff = get_majority(x,f)

if ~exist('f','var')
    f = [2,3];
end

med = median(x);
mn = mean(x);

figure(30)
hist = histogram(x);
histo = [hist.BinEdges',[hist.Values';1]];
close(figure(30))

histo2 = sortrows(histo,2,'descend');

mx = histo2(1,1);
mx_pos = find(histo(:,1)==mx);

if mn>=med
    try
        lim_low = histo(mx_pos-f(1),1);
    catch
        lim_low = histo(1,1);
    end
    try
        lim_high = histo(mx_pos+f(2),1);
    catch
        lim_high = histo(end,1);
    end
else
    try
        lim_low = histo(mx_pos-f(2),1);
    catch
        lim_low = histo(1,1);
    end
    try
        lim_high = histo(mx_pos+f(1),1);
    catch
        lim_high = histo(end,1);
    end
end

cutoff = [lim_low,lim_high];