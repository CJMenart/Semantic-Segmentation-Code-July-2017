function ds = create_samediff_traindata(DatasetHomeDir,ModelSubDir,spVersion,examplesPerImage,show)
% Read up the hypercolumns, labelings and ground-truth segmentations for a
% training dataset, and save out the pairwise examples needed to train a
% classifier on whether segments are the same or a different class. Saves
% everything as a datastore that you can read into memory using a tall
% array.
% WARNING: Has not been tested since updating filenames and whatnot

%Settings
if ~exist('examplesPerImage','var')
    examplesPerImage = 500;
end
split = 'val';
numSameEx = floor(examplesPerImage/2);
numDiffEx = examplesPerImage-numSameEx;
ProbSubDir = strcat(ModelSubDir,'Prob/');
HCSubDir = strcat(ModelSubDir, 'HC/');
SameDiffSubDir = strcat(ModelSubDir,sprintf('SameDiffClassification_%04d/',spVersion));
TrainingExampleSubDir = strcat(SameDiffSubDir,'TrainingExamples/');
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

display('Loading ground truth');
fname = strcat(DatasetHomeDir,split,sprintf('_spGT_%04d',spVersion));
spGT = load(fname); %split_spGT
f = fieldnames(spGT);
spGT = spGT.(f{1});

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
    
    display('Loading segmentations');
    fname = strcat(DatasetHomeDir,split,sprintf('_spIm_%04d',spVersion));
    spIm = load(fname); %split_spIm
    f = fieldnames(spIm);
    spIm = spIm.(f{1});
    
    figA = figure;
end

%-----------------------------

trainingExamplesAccumulated = [];
folderNames = {}; %used to build datastore at end of func
for ImgNum = imRange
%     % load Probabilities for each (super)pixel
%     fname = strcat(DatasetHomeDir, ProbSubDir, split, sprintf('_%04d_prob_%04d.mat',ImgNum,spVersion));
%     if ~exist(fname,'file')
%         continue;
%     end
%     load(fname); % loads 'Prob'
%     % get best class per superpixel
%     origBestClass = vec2ind(Prob)'; 
    origBestClass = spMaxProb{ImgNum};
    
    fname = strcat(DatasetHomeDir, HCSubDir, split, sprintf('_%06d_hc_%04d.mat',ImgNum,spVersion));  
    load(fname); %loads 'spHC'
    
    % count up each site combination of ground truth x classifier label
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
    end
    gtClassesPresent = find(sum(siteCounts,2)>0);
    % We can only do 'same' examples if there are at least two examples of
    % a class
    gtClassesMultiPresent = find(sum(siteCounts,2)>=2); 
    %do nothing with unlabeled or insufficiently labeled im
    if length(gtClassesPresent)<2 && isempty(gtClassesMultiPresent)
        continue;
    end
    
    trainingExamples = zeros(0,'single');
    for ex = 1:examplesPerImage
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
            intensIm = segIm;
            intensIm(:) = 1; %paint it black
            if isSameEx %take it back
                for site = 1:numSites
                    if spGT{ImgNum}(site) == class
                        if origBestClass(site) == class
                            intensIm = color_im_region(intensIm,[1,0,0],spIm{ImgNum}==site);
                        else
                            intensIm = color_im_region(intensIm,...
                                [0,0,double(siteCounts(class,origBestClass(site)))/(numIncorrect)],spIm{ImgNum}==site);
                        end
                    end
                end
            else
                for site = 1:numSites
                    if spGT{ImgNum}(site) == diffPair(1)
                        if origBestClass(site) ~= diffPair(2)
                            intensIm = color_im_region(intensIm,[1,0,0],spIm{ImgNum}==site);
                        else
                            intensIm = color_im_region(intensIm,...
                                [0,0,double(siteCounts(diffPair(1),origBestClass(site))/(numIncorrect))],spIm{ImgNum}==site);
                        end
                    elseif spGT{ImgNum}(site) == diffPair(2)
                        if origBestClass(site) ~= diffPair(1)
                            intensIm = color_im_region(intensIm,[1,0.7,0],spIm{ImgNum}==site);
                        else
                            intensIm = color_im_region(intensIm,...
                                double(siteCounts(diffPair(2),origBestClass(site)))/(numIncorrect)*[1,0,1],spIm{ImgNum}==site);
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
    
    trainingExamplesAccumulated = vertcat(trainingExamplesAccumulated,trainingExamples);
    
    if size(trainingExamplesAccumulated,1)> 100000 || ImgNum == imRange(end)
        %save hypercolumns as tall file
        tallEx = tall(trainingExamplesAccumulated);
        fname = [DatasetHomeDir TrainingExampleSubDir split '_' num2str(ImgNum,'%06d') '_examples'];
        write(fname,tallEx);
        folderNames{ImgNum} = fname;
        trainingExamplesAccumulated = [];
    end
end
    
ds = datastore(folderNames);
save([DatasetHomeDir,TrainingExampleSubDir,'trainingExampleDatastore'],'ds');

if (show)
    close figA;
end
fprintf('Done creating training examples\n.');

end