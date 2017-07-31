function [res] = getFullVGG(VGG, Im)

% make double
Im = double(Im);

% check data range, make sure 0-255, not 0-1
if(max(max(max(Im)))<=1)
    Im=Im*255;
end
    
% subtract mean RGB from each pixel
Im = bsxfun(@minus, Im, VGG.meta.normalization.averageImage);

% make single
Im = single(Im);

% pass image through VGG
res = vl_simplenn(VGG, Im);

end