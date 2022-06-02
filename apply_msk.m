function I_out = apply_msk(RGB,BW)

I_out = RGB;
I_out(repmat(~BW,[1 1 3])) = 0;