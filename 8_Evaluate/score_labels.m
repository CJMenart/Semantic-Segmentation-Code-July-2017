function [jaccardMean,jaccardWeighted,jaccardPerClass,accOverall,accPerClass,meanAcc] = ...
    score_labels(labelSet,segmentedIms,groundTruths,numClasses,show)

    if ~exist('show','var')
        show=false;
    end
    if show
        figA = figure;
    end
    
    andAreas = zeros(numClasses,1);
    orAreas = andAreas;
    trueAreas = andAreas;
    numIms = length(labelSet);
    
    for i = 1:numIms
        [areaAnd,areaOr,areaTrue] = evaluate_label(labelSet{i},segmentedIms{i},...
            groundTruths{i},numClasses);
        andAreas = andAreas + areaAnd;
        orAreas = orAreas + areaOr;
        trueAreas = trueAreas + areaTrue;
        
        if show
            figure(figA);

            subplot(1,2,1);
            imagesc(groundTruths{i});
            caxis([1 numClasses]);
            colormap colorcube;
            axis('image');
            title('Ground Truth');
            
            subplot(1,2,2);
            imagesc(graph_image_labeling(segmentedIms{i},labelSet{i}));
            caxis([1 numClasses]);
            colormap colorcube;
            axis('image');
            title('Final Classification');
            
            pause(0.1);
        end
    end

    % now to compute scores
    jaccardPerClass = zeros(numClasses,1) - 1;
    accPerClass = jaccardPerClass;

    orAreas(orAreas == 0) = 0.0000001;
    trueAreas(trueAreas == 0) = 0.0000001;
    
    jaccardPerClass = andAreas ./ orAreas;
    accPerClass = andAreas ./ trueAreas;
    jaccardMean = mean(jaccardPerClass);
    jaccardWeighted = sum(jaccardPerClass.*trueAreas) / sum(trueAreas);
    accOverall = sum(accPerClass.*trueAreas) / sum(trueAreas);
    meanAcc = sum(accPerClass) / numClasses;
    
    if show
        close(figA);
    end
end