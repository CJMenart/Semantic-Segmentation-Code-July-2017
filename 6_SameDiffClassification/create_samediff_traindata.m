function ds = create_samediff_traindata(DatasetHomeDir,ModelSubDir,settings)
% Read up the hypercolumns, labelings and ground-truth segmentations for a
% training dataset, and save out the pairwise examples needed to train a
% classifier on whether segments are the same or a different class. Saves
% everything as a datastore that you can read into memory using a tall
% array.
% WARNING: Has not been tested since updating filenames and whatnot

%Settings
split = settings.split;
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

trainingExamplesAccumulated = [];
folderNames = {}; %used to build datastore at end of func
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
    
    trainingExamplesAccumulated = vertcat(trainingExamplesAccumulated,trainingExamples);
    
    if size(trainingExamplesAccumulated,1)> 100000 || ImgNum == imRange(end)
        %save hypercolumns as tall file
        tallEx = tall(trainingExamplesAccumulated);
        fname = [settings.trainExampleDir split '_' num2str(ImgNum,'%06d') '_examples'];
        write(fname,tallEx);
        folderNames{ImgNum} = fname;
        trainingExamplesAccumulated = [];
    end
end
    
ds = datastore(folderNames);
save([DatasetHomeDir,settings.trainExampleDir,'trainingExampleDatastore'],'ds');

if (show)
    close figA;
end
fprintf('Done creating training examples\n.');

end