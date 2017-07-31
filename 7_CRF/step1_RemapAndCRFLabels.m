function [] = step1_RemapAndCRFLabels(DatasetHomeDir,ModelSubDir,spVersion,lambda,neighborhoodRadius)
% A version of CRFLabels which takes the classifier probabilities, loads up
% the mapping matrices, and performs re-mapping of probabilities on the fly
% before CRF 
%TODO: Write

if ~exist('radius','var')
    neighborhoodRadius = 1;
end

SameDiffSubDir = strcat(DatasetHomeDir,ModelSubDir,sprintf('SameDiffClassification_%04d/',spVersion));
TestingExampleSubDir = strcat(SameDiffSubDir,'Test Image Pairs/');
ProbSubDir = strcat(ModelSubDir,'Prob/');
split = 'test';
finalResultDir = strcat(DatasetHomeDir,ModelSubDir,'Final Labels/');
mkdir_safe(finalResultDir);
%LOAD
%-----------------------------
display('Loading metadata');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads metaData
numTraining = metaData.numTrain;
numTesting = metaData.numTest;
numClasses = metaData.numClasses;

display('Loading segmentations');
fname = strcat(DatasetHomeDir,split,sprintf('_spIm_%04d',spVersion));
spIm = load(fname); %split_spIm
f = fieldnames(spIm);
spIm = spIm.(f{1});

%-----------------------------

disp('APPLYING CRF OPTIMIZATION TO IMAGE LABELS');

for ImgNum = 1:numTesting
    
    % get edge structure
    adjacencies = neighbors_from_segmentation(spIm{ImgNum},neighborhoodRadius);
     
    %get pairwise costs from same-different classification
    samediffAnswers = load([TestingExampleSubDir sprintf('samediff_testdat_%06d_results.txt',ImgNum)]);
    clear edgeIndices;
    [edgeIndices(:,1),edgeIndices(:,2)] = find(adjacencies);
    edgeIndices = sort(edgeIndices,2);
    edgeIndices = unique(edgeIndices,'rows');
    pairwiseSameProb = zeros(size(adjacencies));
    for edge = 1:size(edgeIndices,1)
        pairwiseSameProb(edgeIndices(edge,1),edgeIndices(edge,2)) = samediffAnswers(edge);
    end
    pairwiseSameProb = pairwiseSameProb + pairwiseSameProb';
    
    %load score map and remap to obtain unary costs
    fname = strcat(DatasetHomeDir,ProbSubDir,sprintf('test_%06d_prob',ImgNum);
    load(fname); % loads 'Prob'
    ***
    
    unaryClassProb = Prob;
    
    %call a specific solver to minimize the costs
    labels = meanInferenceSolver(unaryClassProb,pairwiseSameProb,adjacencies,lambda);
    
    fname = strcat(finalResultDir, sprintf('test_%06d_labels_%04d',ImgNum,spVersion));
    save(fname,'labels');
end
    
fprintf('Done\n.');


end