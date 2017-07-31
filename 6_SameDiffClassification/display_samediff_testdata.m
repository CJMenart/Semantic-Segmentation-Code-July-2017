function display_samediff_testdata(DatasetHomeDir,ModelSubDir,settings)
% Read up the labelings and image for a
% test dataset, read up the same-different answers computed by the neural
% network to show them!
% WARNING: Not tested since updating filenames and whatnot

%Settings
split='test';
SameDiffSubDir = strcat(ModelSubDir,sprintf('SameDiffClassification_%04d/',settings.spVersion));
TestingExampleSubDir = strcat(SameDiffSubDir,'Test Image Pairs/');
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

display('Loading segmentations');
fname = strcat(DatasetHomeDir,split,sprintf('_spIm_%04d',settings.spVersion));
spIm = load(fname); %split_spIm
f = fieldnames(spIm);
spIm = spIm.(f{1});

switch(split)
    case 'train'
        numIms = numTraining;
    case 'test'
        numIms = numTesting;
    otherwise
        error('Unrecognized split!');
end

figA = figure;
for ImgNum = 1:numIms
    
    img = load([ImDir,split,sprintf('_img_%06d',ImgNum)]);
    img = img.rgbIm;
    
    neighbors = neighbors_from_segmentation(spIm{ImgNum},settings.neighborhoodRadius);
    clear exampleIndices;
    [exampleIndices(:,1),exampleIndices(:,2)] = find(neighbors);
    exampleIndices = sort(exampleIndices,2);
    exampleIndices = unique(exampleIndices,'rows');
    numExamples = size(exampleIndices,1);
    
    adjacencies = neighbors_from_segmentation(spIm{ImgNum},1);
    clear adjIndices;
    [adjIndices(:,1),adjIndices(:,2)] = find(adjacencies);
    adjIndices = sort(adjIndices,2);
    adjIndices = unique(adjIndices,'rows');
    
    samediffAnswers = load([DatasetHomeDir TestingExampleSubDir sprintf('samediff_testdat_%06d_results.txt',ImgNum)]);
    
    for ex = 1:numExamples
        if ~ismember(exampleIndices(ex,:),adjIndices,'rows')
            continue; %only draw lines between adjacent sites
        end
        
        c1 = regionprops(spIm{ImgNum}==exampleIndices(ex,1),'centroid');
        c2 = regionprops(spIm{ImgNum}==exampleIndices(ex,2),'centroid'); 
        
        sameConf = samediffAnswers(ex);
        color = [255*(1-sameConf) 255*sameConf 0];
        
        img = draw_line(img,round(c1.Centroid),round(c2.Centroid),color);
    end
    
    subplot(1,1,1);
    imagesc(img);
    title(sprintf('Image %d',ImgNum));
    pause(0.5);
end
    
fprintf('Done\n.');

end