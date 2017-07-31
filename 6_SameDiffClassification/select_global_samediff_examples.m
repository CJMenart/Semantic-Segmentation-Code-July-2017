% WARNING: UNUSED

function [trainingExamples] = select_global_samediff_examples(spIm,origBestClass,...
                                spGT,spHC,numClasses,numExamples,img,figA)
%An unfinished code refactor which was decomposing the example-selection
%into smaller, more reasonable functions. Only the local version has been
%tested yet, so this one might have a couple lingering bugs
%Christopher Menart, 7/19/17
                            
show = (exist('img','var') && exist('figA','var'));
trainingExamples = zeros(0,'single');
numSameEx = floor(settings.examplesPerImage/2);
numDiffEx = settings.examplesPerImage-numSameEx;

% count up each site combination of ground truth x classifier label
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
end
gtClassesPresent = find(sum(siteCounts,2)>0);
% We can only do 'same' examples if there are at least two examples of
% a class
gtClassesMultiPresent = find(sum(siteCounts,2)>=2); 
%do nothing with unlabeled or insufficiently labeled im
if length(gtClassesPresent)<2 && isempty(gtClassesMultiPresent)
    return;
end

for ex = 1:settings.examplesPerImage
    %choose sites to use
    %first half should be examples of where the two sites are same
    exampleIndices = [];
    if (ex <= numSameEx && ~isempty(gtClassesMultiPresent)) || length(gtClassesPresent) < 2
        isSameEx = true;
        sameExPerClass = numSameEx/length(gtClassesMultiPresent);
        class = gtClassesMultiPresent(min(ceil(ex/sameExPerClass),length(gtClassesMultiPresent)));
        numCorrect = siteCounts(class,class);
        numIncorrect = sum(siteCounts(class,:)) - numCorrect;

        %pick segment 1
        if numIncorrect == 0 || (rand() > 0.5 && numCorrect > 0)
            exampleIndices(1) = siteIndices{class,class}(randi(numCorrect));
        else
            incorrectIndices = horzcat(siteIndices{class,1:class-1},siteIndices{class,class+1:end});
            exampleIndices(1) = incorrectIndices(randi(numIncorrect));
        end

        %pick segment 2
        exampleIndices(2)=exampleIndices(1);
        while(exampleIndices(1)==exampleIndices(2))
            if numIncorrect == 0 || (rand() > 0.5 && numCorrect > 0)
                exampleIndices(2) = siteIndices{class,class}(randi(numCorrect));
            else
                incorrectIndices = horzcat(siteIndices{class,1:class-1},siteIndices{class,class+1:end});
                exampleIndices(2) = incorrectIndices(randi(numIncorrect));
            end
        end            
        trainTarget = 1;
    else %second half of examples should be of different sites, if possible
        isSameEx = false;
        allDiffPairs = nchoosek(gtClassesPresent,2);
        diffExPerClassPair = numDiffEx/size(allDiffPairs,1);
        diffPair = allDiffPairs(min(ceil((ex-numDiffEx+1)/diffExPerClassPair),size(allDiffPairs,1)),:);
        for segment = 1:2
            % we only count as 'incorrect', examples which label class
            % A as class B when collecting 'different' examples
            classA = diffPair(segment);
            classB = diffPair(3-segment);
            numIncorrect = siteCounts(classA,classB);
            numCorrect = sum(siteCounts(classA,:)) - numIncorrect;
            if numIncorrect == 0 || (rand() > 0.5 && numCorrect > 0)
                correctIndices = horzcat(siteIndices{classA,1:classB-1},siteIndices{classA,classB+1:end});
                exampleIndices(segment) = correctIndices(randi(numCorrect));                    
            else %incorrect case
                exampleIndices(segment) = siteIndices{classA,classB}(randi(numIncorrect));
            end
        end
        trainTarget = 0;
    end

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
        intensIm = segIm;
        intensIm(:) = 1; 
        if isSameEx 
            for site = 1:numSites
                if spGT{ImgNum}(site) == class
                    if origBestClass(site) == class
                        intensIm = color_im_region(intensIm,[1,0,0],spIm==site);
                    else
                        intensIm = color_im_region(intensIm,...
                            [0,0,double(siteCounts(class,origBestClass(site)))/(numIncorrect)],spIm==site);
                    end
                end
            end
        else
            for site = 1:numSites
                if spGT{ImgNum}(site) == diffPair(1)
                    if origBestClass(site) ~= diffPair(2)
                        intensIm = color_im_region(intensIm,[1,0,0],spIm==site);
                    else
                        intensIm = color_im_region(intensIm,...
                            [0,0,double(siteCounts(diffPair(1),origBestClass(site))/(numIncorrect))],spIm==site);
                    end
                elseif spGT{ImgNum}(site) == diffPair(2)
                    if origBestClass(site) ~= diffPair(1)
                        intensIm = color_im_region(intensIm,[1,0.7,0],spIm==site);
                    else
                        intensIm = color_im_region(intensIm,...
                            double(siteCounts(diffPair(2),origBestClass(site)))/(numIncorrect)*[1,0,1],spIm==site);
                    end
                end
            end
        end
        imshow(intensIm);
        hold on
        plot(c1.Centroid(1),c1.Centroid(2),'w+','MarkerSize',30);
        plot(c2.Centroid(1),c2.Centroid(2),'w+','MarkerSize',30);
        hold off
        if isSameEx
            title(sprintf('Picking ''Same'' Examples of Class %d',class));
        else
            title(sprintf('Picking ''Different'' Examples of Class %d and %d',diffPair(1),diffPair(2)));
        end

        pause(0.1);
    end

    %collect out hypercolumns for given indices
    trainingExamples(ex,:) = horzcat(trainTarget,spHC(:,exampleIndices(1))',...
        spHC(:,exampleIndices(2))');
end

end