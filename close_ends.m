function BW1 = close_ends(BW)
sz = size(BW);

ffn = find(BW(1,:)==1);
if length(ffn)>2
    x1 = min(ffn);
    x2 = max(ffn);
    BW(1,x1:x2)=1;
end

ffn2 = find(BW(sz(1),:)==1);
if length(ffn2)>2
    x3 = min(ffn2);
    x4 = max(ffn2);
    BW(sz(1),x3:x4)=1;
end
BW1 = imfill(BW,'holes');