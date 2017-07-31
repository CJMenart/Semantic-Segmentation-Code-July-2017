function ds = create_local_samediff_traindata(DatasetHomeDir,ModelSubDir,spVersion,examplesPerImage,neighborhoodRadius,show)
% Read up the hypercolumns, labelings and ground-truth segmentations for a
% training dataset, and save out the pairwise examples needed to train a
% classifier on whether segments are the same or a different class. Saves
% everything as a datastore that you can read into memory using a tall
% array.
% This function differs from create_samediff_trandata in that it only picks
% as training examples segments that are adjacent to one another. The
% reasoning behind this is that that's the sort of thing we're actually
% going to be using this network for, so we don't want the network to just
% learn that adjacent segments with the same high-level hypercolumns are
% often the same class. Assuming that's a thing. Anyway, this also
% necessitates some changes in the hierarchy here.

%Settings
if ~exist('neighborhoodRadius','var')
    neighborhoodRadius = 1; %max number of pixels from one site to another
end
if ~exist('examplesPerImage','var')
    examplesPerImage = 500;
end
split = 'train';
numSameEx = floor(examplesPerImage/2);
numDiffEx = examplesPerImage-numSameEx;
ProbSubDir = strcat(ModelSubDir,'Prob/');
HCSubDir = strcat(DatasetHomeDir, 'HC/');
SameDiffDir = strcat(DatasetHomeDir,ModelSubDir,sprintf('SameDiffClassification_%04d/',spVersion));
TrainingExampleDir = strcat(SameDiffDir,'TrainingExamples/');
if strcmp(split,'test')
    ImDir = strcat(DatasetHomeDir,'TestImgs/');
else
    ImDir = strcat(DatasetHomeDir,'TrainImgs/');
end

if ~exist('show','var')
    show=false;
end

%LOAD
%-----------------------------
display('Loading metadata');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads metaData
numTraining = metaData.numTrain;
numTesting = metaData.numTest;
numClasses = metaData.numClasses;
whichTrainAreVal = metaData.whichTrainAreVal;

switch(split)
    case 'train'
        numIms = numTraining;
        imRange = 1:numTraining;
    case 'test'
        numIms = numTesting;
        imRange = 1:numTesting;
    case 'val' %val is a subset of train
        numIms = length(whichTrainAreVal);
        imRange = whichTrainAreVal;
        split = 'train';
    otherwise
        error('Unrecognized split!');
end

display('Loading segment ground truth');
fname = strcat(DatasetHomeDir,split,sprintf('_spGT_%04d',spVersion));
spGT = load(fname); %split_spGT
f = fieldnames(spGT);
spGT = spGT.(f{1});

display('Loading segmentations');
fname = strcat(DatasetHomeDir,split,sprintf('_spIm_%04d',spVersion));
spIm = load(fname); %split_spIm
f = fieldnames(spIm);
spIm = spIm.(f{1});
    
display('Loading ''Best Softmax''');
fname = strcat(DatasetHomeDir,ModelSubDir,split,sprintf('_spMaxProb_%04d',spVersion));
spMaxProb = load(fname);
f = fieldnames(spMaxProb);
spMaxProb = spMaxProb.(f{1});

if (show)
%     display('Loading images');
%     fname = strcat(DatasetHomeDir,split,'_imgs');
%     imgs = load(fname); %split_imgs
%     f = fieldnames(imgs);
%     imgs = imgs.(f{1});
        
    display('Loading ground truth');
    fname = strcat(DatasetHomeDir,split,sprintf('_pixeltruth',spVersion));
    pixeltruth = load(fname); %split_spIm
    f = fieldnames(pixeltruth );
    pixeltruth = pixeltruth .(f{1});
    
    figA = figure;
end

%-----------------------------


folderNames = {}; %used to build datastore at end of func
trainingExamplesAccumulated = [];
for ImgNum = imRange
    
    %Version of code which loads up actual probability files to get best
    %softmax.
%     % load Probabilities for each (super)pixel
%     fname = strcat(DatasetHomeDir, ProbSubDir, split, sprintf('_%04d_prob_%04d.mat',ImgNum,spVersion));
%     if ~exist(fname,'file')
%         continue;
%     end
%     load(fname); % loads 'Prob'
%     % get best class per superpixel
%     origBestClass = vec2ind(Prob)'; 
    origBestClass = spMaxProb{ImgNum};

    fname = strcat(HCSubDir, split, sprintf('_%06d_hc_%04d.mat',ImgNum,spVersion));  
    load(fname); %loads 'spHC'
    
    % find out the 'nearby' sites for each site
    neighborSites = neighbors_from_segmentation(spIm{ImgNum},neighborhoodRadius);

    % count up each site combination of classifier x ground truth label
    % Also, Certain classes, in wonky pictures, may be unable to produce 'same'
    % or 'different' pairs. We figure that out here. It's tedious. It was
    % tedious to write, it's tedious to run.
    neighborClasses = zeros(numClasses);
    siteCounts = zeros(numClasses);
    siteIndices = cell(numClasses);
    numSites = length(origBestClass);
    for site = 1:numSites
        gt = spGT{ImgNum}(site);
        if gt==0
            continue
        end
        cl = origBestClass(site);
        siteCounts(gt,cl) = siteCounts(gt,cl) + 1;
        siteIndices{gt,cl} = [siteIndices{gt,cl} site];
        
        for site2 = find(neighborSites(site,:))
            gt2 = spGT{ImgNum}(site2);
            if gt2~=0
                neighborClasses(gt,gt2)=1;
            end 
        end
    end
    gtClassesPresent = find(sum(siteCounts,2)>0);
    gtSameClassesPresent = find(diag(neighborClasses));
    gtDiffClassesPresent = find(sum(neighborClasses,2) > diag(neighborClasses));
        
    if isempty(gtSameClassesPresent) && isempty(gtDiffClassesPresent)
        continue;
    end    
    
    trainingExamples = zeros(0,'single');
    for ex = 1:examplesPerImage
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
            img = imread(ImDir,split,sprintf('_img_%06d',ImgNum));
            
            c1 = regionprops(spIm{ImgNum}==exampleIndices(1),'centroid');
            c2 = regionprops(spIm{ImgNum}==exampleIndices(2),'centroid');
            
            figure(figA);
            subplot(1,3,1);
            segIm = segImage(im2double(img),spIm{ImgNum});
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
    trainingExamplesAccumulated = vertcat(trainingExamplesAccumulated,trainingExamples);
    
    if size(trainingExamplesAccumulated,1) > 10000 || ImgNum == imRange(end)
        %save hypercolumns as tall file
        tallEx = tall(trainingExamplesAccumulated);
        fname = [TrainingExampleDir split '_' num2str(ImgNum,'%06d') '_examples'];
        write(fname,tallEx);
        folderNames{end+1} = fname;
        trainingExamplesAccumulated = [];
    end
end
    
ds = datastore(folderNames,'Type','tall');
save([TrainingExampleDir,'trainingExampleDatastore'],'ds');

if (show)
    close figA;
end
fprintf('Done creating training examples\n.');

end