%Determine the neighbors of each segment
%got some help from 
%http://stackoverflow.com/questions/31517912/identify-the-adjacent-pixels-in-matlab

function neighbors = neighbors_from_segmentation(segmentedIm,radius)
    %radius: maxmum distance to another site for it to be considered a
    %neighbor. Default is 1, which only counts sites that touch
 
    numSegments = max(max(segmentedIm));
    neighbors = zeros(numSegments,'double');

    if ~exist('radius','var')
        radius = 1;
    end
    AdjacencyMask = ones(2*radius+1);

    for seg = 1:numSegments  %parfor temporarily disabled
        segNeighbors = neighbors(seg,:);
        adjacentPixelInds = conv2(double(segmentedIm == seg), AdjacencyMask,...
            'same') > 0;
        for neighbor = unique(segmentedIm(adjacentPixelInds))
            %for now, no specal weight, but we can make this cumulative to give
            %more weight to neighbors that are 'more adjacent'
            segNeighbors(1,neighbor) = 1; 
        end
        neighbors(seg,:) = segNeighbors;
    end
    for seg = 1:numSegments
        neighbors(seg,seg) = 0; 
    end

end