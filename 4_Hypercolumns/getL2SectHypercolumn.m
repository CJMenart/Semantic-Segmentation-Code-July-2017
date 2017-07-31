function [ hc ] = getL2SectHypercolumn(res, r, c, nrows, ncols, whichVGGHCLayersToUse)

hc = [];

% get data from each desired layer
for layerInd = 1:length(whichVGGHCLayersToUse)
    
    % pick layer
    layer = whichVGGHCLayersToUse(layerInd);
    
    % get output of that layer
    output = gather(res(layer+1).x);
    
    % res output size
    out_r = double(size(output,1));
    out_c = double(size(output,2));
                
    % Get integer scale factor (pow-2) of expansion for feature map to
    % original image.
    % Make sure to use round to handle machine precision (keep only 5
    % decimals)
    int_sr = 2^( ceil( round(log2(double(nrows)/out_r), 5) ) );
    int_sc = 2^( ceil( round(log2(double(ncols)/out_c), 5) ) );
    
    % get output location from original r,c
    int_r = ceil( round(double(r)/int_sr, 5) );
    int_c = ceil( round(double(c)/int_sc, 5) );
    
    % get features at this location in the layer
    features = squeeze(output(int_r, int_c,:));
    
    % L2-norm this portion of the features
    nm=norm(features,2);
    if(nm>0)
        features = features/nm;
    end
    
    % append to column
    hc = [hc; features(:)];
    
end


end
