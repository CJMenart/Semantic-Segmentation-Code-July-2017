% Assigns computes 'areas' for to an image labeling
% that you pass in. These areas are used by score_labels to compute various
% scores.

% INPUTS:
% labels: a set of class assignments, numSegments in length
% segmentedIm: The image segmentation to map the labels too
% groundTruth: a groundTruth image of labels, same size as original image

% OUTPUTS:
% areaAnd: The area (in pixels) labeled as a certain class by both the
% labeler and ground truth.
% areaOr: Same as the above, but or instead of and. These two quantities
% areused for computing the Jaccard score.
% pixelAccuracy: The area, in pixels, actually labeled as a given class in
% the ground truth.
%
% From these three outputs you can compute the Jaccard index and pixel
% accuracy.

function [areaAnd,areaOr,areaTrue] = evaluate_label(labels,segmentedIm,...
    groundTruth,numClasses)

    labeling =  graph_image_labeling(segmentedIm, labels);
    
    % unlabeled regions of the image are absolute don't-cares and should
    % not be counted for anything.
    labeling = labeling(groundTruth > 0);
    groundTruth = groundTruth(groundTruth > 0);
    
    areaAnd = zeros(numClasses,1);
    areaOr = areaAnd;
    areaTrue = areaAnd;
    
    for c = 1:numClasses
        areaAnd(c) = numel(labeling(labeling == c & groundTruth == c));
        areaOr(c) = numel(labeling(labeling == c | groundTruth == c));
        areaTrue(c) = numel(groundTruth(groundTruth == c));
    end
end