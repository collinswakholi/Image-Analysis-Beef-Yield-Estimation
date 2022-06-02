% Range 정규화 보정 스펙트럼
function [Spectral] = Range_normalize (x)

[XS1 YS1] = size(x);
% Max_spectra = max(x(i,:));
% Min_spectra = min(x(i,:));
Spectral = zeros(XS1, YS1);

for i = 1:XS1;
    Max_spectra = max(x(i,:));
    Min_spectra = min(x(i,:));
    Spectral(i,:) = x(i,:)'/(Max_spectra-Min_spectra);
end