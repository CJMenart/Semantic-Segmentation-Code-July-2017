%%% graph_image_labeling
%%% Takes a set of ordered labels, aka the output of a label_image
%%% function, and a mori segmented image, and imposes the labels on the
%%% segments so that you can visualize the labeling.
%%%
%%% set the optional parameter 'show' to true in order to plot it

%TODO: An alternate mode where you can superimpose 'ground truth?"

function visualization = graph_image_labeling(segmentedIm, labels, show)
    
    if nargin < 3
       show = false; 
    end

    visualization = segmentedIm;
    numSegments = length(labels);

    if numSegments ~= max(max(segmentedIm))
        error('Different number of segments in labels and superpixels!');
    end
    
    for segment = 1:numSegments
        visualization(segmentedIm==segment) = labels(segment);
    end
    
    if show
        imshow(visualization);
    end
end