function create_samediff_testdata(DatasetHomeDir,ModelSubDir,settings)
% Read up the hypercolumns, labelings and ground-truth segmentations for a
% test dataset, and save out the pairwise examples needed to test the
% classifier. We don't need to shuffle anything, so we're just going to
% write it out straight to CSV. The pattern for sites is upper-diagonal:
% first you have site 1, connected to everything that touches it, then site 2,
% connected to everything after 2 that touches it, then site 3...etc. We will have to note
% this when reading the answers as well as writing data.

%Settings
split = 'test';
ProbSubDir = strcat(ModelSubDir,'Prob/');
HCDir = strcat(DatasetHomeDir, 'HC/');

%LOAD
%-----------------------------
display('Loading metadata');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads metaData
numTraining = metaData.numTrain;
numTesting = metaData.numTest;
numClasses = metaData.numClasses;

display('Loading ground truth');
fname = strcat(DatasetHomeDir,split,sprintf('_spGT_%04d',settings.spVersion));
spGT = load(fname); %split_spGT
f = fieldnames(spGT);
spGT = spGT.(f{1});

display('Loading segmentations');
fname = strcat(DatasetHomeDir,split,sprintf('_spIm_%04d',settings.spVersion));
spIm = load(fname); %split_spIm
f = fieldnames(spIm);
spIm = spIm.(f{1});

display('Loading data mean');
fname = strcat([settings.sameDiffDir 'meanVec']);
load(fname); % loads 'meanVec'    

%-----------------------------

switch(split)
    case 'train'
        numIms = numTraining;
    case 'test'
        numIms = numTesting;
    otherwise
        error('Unrecognized split!');
end
folderNames = {}; %used to build datastore at end of func
for ImgNum = 1:numIms

    fname = strcat(HCDir, split, sprintf('_%06d_hc_%04d.mat',ImgNum,settings.spVersion));  
    load(fname); %loads 'spHC'

    %where data will be saved.
    fname = [settings.testDataDir 'samediff_testdat_' num2str(ImgNum,'%06d') '.csv'];
    if exist(fname,'file')
        fprintf('Found file %s already, skippping...\n',fname);
        continue;
    end
    
    testingExamples = zeros(0,'single');
    adjacencies = neighbors_from_segmentation(spIm{ImgNum},settings.neighborhoodRadius);
    clear exampleIndices;
    [exampleIndices(:,1),exampleIndices(:,2)] = find(adjacencies);
    exampleIndices = sort(exampleIndices,2);
    exampleIndices = unique(exampleIndices,'rows');
    
    numExamples = size(exampleIndices,1);
    testingExamples = horzcat(-1*ones(numExamples,1,'single'),spHC(:,exampleIndices(:,1))',...
        spHC(:,exampleIndices(:,2))');
    %mean-subtraction
    testingExamples(:,2:end) = testingExamples(:,2:end) - repmat(meanVec,numExamples,2);
    
    %save hypercolumns
    csvwrite(fname,testingExamples);
end
    
fprintf('Done\n.');

end