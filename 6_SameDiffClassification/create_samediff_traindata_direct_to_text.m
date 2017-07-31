function [] = create_samediff_traindata_direct_to_text(DatasetHomeDir,ModelSubDir,settings)
% Read up the hypercolumns, labelings and ground-truth segmentations for a
% training dataset, and save out the pairwise examples needed to train a
% classifier on whether segments are the same or a different class. Saves
% everything as a datastore that you can read into memory using a tall
% array.
% WARNING: Has not been tested since updating filenames and whatnot
%This version of the code goes directly to text csv files. This may reduce
%accuracy, especially if the amount of pooling in the later part of
%training is low, and it requires you to read all the hypercolumns to find
%the mean of training data. But it avoids having to touch parallel pool in
%case you're having some issues with that.

%TODO: Write

%Settings
split = settings.split;
numSameEx = floor(settings.examplesPerImage/2);
numDiffEx = settings.examplesPerImage-numSameEx;
ProbSubDir = strcat(ModelSubDir,'Prob/');
HCSubDir = 'HC/';
if strcmp(split,'test')
    ImDir = strcat(DatasetHomeDir,'TestImgs/');
else
    ImDir = strcat(DatasetHomeDir,'TrainImgs/');
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
fname = strcat(DatasetHomeDir,split,sprintf('_spGT_%04d',settings.spVersion));
spGT = load(fname); %split_spGT
f = fieldnames(spGT);
spGT = spGT.(f{1});

display('Loading ''Best Softmax''');
fname = strcat(DatasetHomeDir,ModelSubDir,split,sprintf('_spMaxProb_%04d',settings.spVersion));
spMaxProb = load(fname);
f = fieldnames(spMaxProb);
spMaxProb = spMaxProb.(f{1});

display('Loading segmentations');
fname = strcat(DatasetHomeDir,split,sprintf('_spIm_%04d',settings.spVersion));
spIm = load(fname); %split_spIm
f = fieldnames(spIm);
spIm = spIm.(f{1});

if (settings.show)
    figA = figure;
end

%-----------------------------
%compute and save data mean
fprintf('Computing Mean Hypercolumn\n');
meanVec = [];

for ImgNum = imRange
    
fname = strcat(DatasetHomeDir, HCSubDir, split, sprintf('_%06d_hc_%04d.mat',ImgNum,settings.spVersion));  
load(fname); %loads 'spHC'

if isempty(meanVec)
    meanVec = mean(spHC,2); 
else
    meanVec = meanVec + mean(spHC,2);
end
end

meanVec = (meanVec./length(imRange))';
save([settings.sameDiffDir 'meanVec'],'meanVec');
%-----------------------------

%deterministic randomstream in case the process crashes halfway through...
rs = RandStream('mt19937ar','Seed',1492);
shuffled = imRange(randperm(rs,length(imRange)));
valInds = shuffled(1:round(settings.valProp*length(imRange)));
trainNum = 0;
valNum = 0;

for ImgNum = imRange

    origBestClass = spMaxProb{ImgNum};
    
    fname = strcat(DatasetHomeDir, HCSubDir, split, sprintf('_%06d_hc_%04d.mat',ImgNum,settings.spVersion));  
    load(fname); %loads 'spHC'
    
    if settings.show
        img = imread(ImDir,split,sprintf('_img_%06d',ImgNum));
        trainingExamples = select_local_samediff_examples(spIm{ImgNum},origBestClass,spGT{ImgNum},spHC,numClasses,settings,img,figA);
    else
        trainingExamples = select_local_samediff_examples(spIm{ImgNum},origBestClass,spGT{ImgNum},spHC,numClasses,settings);
    end
    
    if isempty(trainingExamples) %some images have no labels and will not be used
        continue;
    end
    
    trainingExamples(:,2:end) = trainingExamples(:,2:end) - repmat(meanVec,settings.examplesPerImage,2);
    
    if ismember(ImgNum,valInds)
        valNum=valNum+1;
        fname = [settings.trainDataDir sprintf('\\samediff_valdat_%d.csv',valNum)];
    else
        trainNum=trainNum+1;
        fname = [settings.trainDataDir sprintf('\\samediff_traindat_%d.csv',trainNum)];
    end
    csvwrite(fname,trainingExamples);
end
    
if (settings.show)
    close figA;
end
fprintf('Done creating training examples\n.');

end