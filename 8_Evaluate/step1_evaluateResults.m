function [] = step1_evaluateResults(DatasetHomeDir, ModelSubDir, spVersion, useDisplay)


%----- 

% directories
ResultsSubDir = strcat(ModelSubDir,'Final Labels/');

display('Loading metadata');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads metaData
numTesting = metaData.numTest;
numClasses = metaData.numClasses;

display('Loading test pixel truth images');  
fname = strcat(DatasetHomeDir, 'test_pixeltruth.mat');
load(fname); % loads test_pixeltruth

% not using actual superpixels
if(spVersion==0)
    useSP=0;
else
    useSP=1;
    
    display('Loading test superpixel images');  
    fname = strcat(DatasetHomeDir, sprintf('test_spIm_%04d.mat',spVersion));
    load(fname); % loads test_spIm
end

%---------------------------------

% load results 
allLabels = {};
for TestImgNum = 1:numTesting
    fname = strcat(DatasetHomeDir, ResultsSubDir, sprintf('test_%06d_labels_%04d.mat',TestImgNum,spVersion));
    load(fname) % loads 'labels'
    allLabels{TestImgNum} = labels;
end


%-------------------------------

  
[jaccardMean,jaccardWeighted,jaccardPerClass,accOverall,accPerClass,meanAcc] = ...
    score_labels(allLabels,test_spIm,test_pixeltruth,numClasses,useDisplay);

%-------------------------------

fprintf('Dataset Result Overall Pixel Accuracy: %.3f\n', accOverall);
fprintf('Dataset Result Mean Pixel Accuracy: %.3f\n', meanAcc);
fprintf('Dataset Result Mean Intersection over Union: %.3f\n', jaccardMean);
fprintf('Dataset Result Frequency-Weighted Intersection over Union: %.3f\n', jaccardWeighted);
fprintf('Dataset Result Accuracy per Class: \n');
disp(accPerClass);
fprintf('Dataset Result Intersection Over Union per Class: \n');
disp(jaccardPerClass);

% save results 
fname = strcat(DatasetHomeDir, ResultsSubDir, 'pixacc_results.mat');
save(fname, 'jaccardMean','jaccardWeighted','jaccardPerClass','accOverall','accPerClass','meanAcc', '-v7.3');

display('Done');
display('===========================');

end
