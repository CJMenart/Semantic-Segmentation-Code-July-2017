function [ spHC ] = getVGGAveL2SectHypercolumn(spIm, res, whichVGGHCLayersToUse)


% number of superpixels
numSP = max(spIm(:));

% scale dims
numRows = size(spIm,1);
numCols = size(spIm,2);

% process each
for i=1:numSP

    % superpixel index
    sp = i;
    
    % get centroid location for this superpixel
    [rows, cols] = find(spIm==sp);
    mean_r = mean(rows);
    mean_c = mean(cols);
    
    % get hypercolumn for location 
    % (NOTE: this performs L2-norm on each section of hypercolumn)
    spHC(:,i) = getL2SectHypercolumn(res, mean_r, mean_c, numRows, numCols, whichVGGHCLayersToUse);
     
end

end