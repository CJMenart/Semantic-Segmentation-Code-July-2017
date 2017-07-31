% WARNING: UNUSED

function [trainingExamples] = select_local_samediff_examples(spIm,origBestClass,...
                                spGT,spHC,numClasses,settings,img,figA)
%Logic of selecting things from images
%Christopher Menart, 7/19/17
                            
show = (exist('img','var') && exist('figA','var'));
trainingExamples = zeros(0,'single');
numSameEx = floor(settings.examplesPerImage/2);
numDiffEx = settings.examplesPerImage-numSameEx;

% find out the 'nearby' sites for each site
neighborSites = neighbors_from_segmentation(spIm,settings.neighborhoodRadius);

% count up each site combination of classifier x ground truth label
% Also, Certain classes, in wonky pictures, may be unable to produce 'same'
% or 'different' pairs. We figure that out here. It's tedious. It was
% tedious to write, it's tedious to run.
neighborClasses = zeros(numClasses);
siteCounts = zeros(numClasses);
siteIndices = cell(numClasses);
numSites = length(origBestClass);
for site = 1:numSites
    gt = spGT(site);
    if gt==0
        continue
    end
    cl = origBestClass(site);
    siteCounts(gt,cl) = siteCounts(gt,cl) + 1;
    siteIndices{gt,cl} = [siteIndices{gt,cl} site];

    for site2 = find(neighborSites(site,:))
        gt2 = spGT(site2);
        if gt2~=0
            neighborClasses(gt,gt2)=1;
        end 
    end
end
gtClassesPresent = find(sum(siteCounts,2)>0);
gtSameClassesPresent = find(diag(neighborClasses));
gtDiffClassesPresent = find(sum(neighborClasses,2) > diag(neighborClasses));

if isempty(gtSameClassesPresent) && isempty(gtDiffClassesPresent)
    return;
end    

for ex = 1:settings.examplesPerImage
    %choose sites to use
    %first half should be examples of where the two sites are same
    exampleIndices = [];
    if (ex <= numSameEx && ~isempty(gtSameClassesPresent)) || isempty(gtDiffClassesPresent)
        isSameEx = true;
        sameExPerClass = numSameEx/length(gtSameClassesPresent);
        class = gtSameClassesPresent(min(ceil(ex/sameExPerClass),length(gtSameClassesPresent)));
        trainTarget = 1;
    else
        isSameEx = false;
        diffExPerClass = numDiffEx/length(gtDiffClassesPresent);
        class = gtDiffClassesPresent(min(ceil((ex-numDiffEx)/diffExPerClass),length(gtDiffClassesPresent)));
        trainTarget = 0;
    end

    numCorrect = siteCounts(class,class);
    numIncorrect = sum(siteCounts(class,:)) - numCorrect;

    %pick segments
    possibleSegment2 = [];
    while isempty(possibleSegment2)
        if numIncorrect == 0 || (rand() > 0.5 && numCorrect > 0)
            exampleIndices(1) = siteIndices{class,class}(randi(numCorrect));
        else
            incorrectIndices = horzcat(siteIndices{class,1:class-1},siteIndices{class,class+1:end});
            exampleIndices(1) = incorrectIndices(randi(numIncorrect));
        end

        neigh = find(neighborSites(exampleIndices(1),:));

        if isSameEx
            possibleSegment2 = neigh(ismember(neigh,horzcat(siteIndices{class,:})));
        else
            possibleSegment2 = neigh(ismember(neigh,horzcat(siteIndices{1:class-1,:},siteIndices{class+1:end,:})));
        end
    end            
    %pick segment 2
    exampleIndices(2) = possibleSegment2(randsample(length(possibleSegment2),1));

    if (show)
        c1 = regionprops(spIm==exampleIndices(1),'centroid');
        c2 = regionprops(spIm==exampleIndices(2),'centroid');

        figure(figA);
        subplot(1,3,1);
        segIm = segImage(im2double(img),spIm);
        imshow(segIm);
        hold on
        plot(c1.Centroid(1),c1.Centroid(2),'w+','MarkerSize',20);
        plot(c2.Centroid(1),c2.Centroid(2),'w+','MarkerSize',20);
        hold off
        title(sprintf('Image %d',ImgNum));

        subplot(1,3,2);
        imagesc(siteCounts);
        title('Counts of each site type');
        xlabel('Classifier Label')
        ylabel('Ground Truth Label');
        axis('square');

        subplot(1,3,3);            
        imagesc(pixeltruth{ImgNum});
        caxis([1 numClasses]);
        hold on
        plot(c1.Centroid(1),c1.Centroid(2),'w+','MarkerSize',30);
        plot(c2.Centroid(1),c2.Centroid(2),'w+','MarkerSize',30);
        hold off
        if isSameEx
            title(sprintf('Picking ''Same'' Examples of Class %d',class));
        else
            title(sprintf('Picking ''Different'' Examples of Class %d',class));
        end

        pause(0.1);
    end

    %collect out hypercolumns for given indices
    trainingExamples(ex,:) = horzcat(trainTarget,spHC(:,exampleIndices(1))',...
        spHC(:,exampleIndices(2))');
end

end