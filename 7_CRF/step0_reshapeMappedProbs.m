function [] = step0_reshapeMappedProbs(DatasetHomeDir,ModelSubDir,spVersion)

% sub directories
RemappedDir = strcat(DatasetHomeDir, ModelSubDir, 'Remapped Probs/');

% load metadata
display('Loading meta-data');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'
numClasses = metaData.numClasses;
numTesting = metaData.numTest;

% testing superpixels
fname = strcat(DatasetHomeDir, sprintf('test_spIm_%04d.mat',spVersion));
load(fname); % loads 'test_spIm'

fname = strcat(DatasetHomeDir, sprintf('test_spGT_%04d.mat',spVersion));
load(fname); % loads 'test_spGT'
        
% process images
for i=1:numTesting

    % get superpixel image
    spIm = test_spIm{i};

    % load softmax probability 
    clear Prob;
    fname = strcat(RemappedDir, sprintf('remappedMat_test%07d',i));
    load(fname); % loads 'remappedMat'
    Prob = reshape(remappedMat,[],numClasses)'; %needs reshaping...
    
    % get/save superpixel max probability
    [Prob, spMaxProb] = assignProbtoSP(Prob, spIm);

    fname = strcat(RemappedDir, sprintf('test_%06d_prob_%04d',i,spVersion));
    save(fname,'Prob');
 
end

display('Done\n');

end