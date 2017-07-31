function [] = step1_writeCityscapesResults(DatasetHomeDir, ModelSubDir, spVersion, useDisplay)
% The cityscapes test annotations are contained on an evaluation server.
% Instead of computing score metrics ourselves, this code simply writes out
% our final annotations in the correct format to be sent to this server.
% Additionally, we are working with images at one-quarter resolution, so we
% have to upsize our segmentations before we write results.
% TODO: Double-check required formatting and test function

%----- 

% directories
ResultsSubDir = strcat(ModelSubDir,'Final Labels/');
SubmissionSubDir = strcat(ModelSubDir,'Submission for Cityscapes Server/');

display('Loading metadata');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads metaData
numTesting = metaData.numTest;
numClasses = metaData.numClasses;
test_imageNames = metaData.test_imageNames;

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

for TestImgNum = 1:numTesting

    fname = strcat(DatasetHomeDir, ResultsSubDir, sprintf('test_%06d_labels_%04d.mat',TestImgNum,spVersion));
    load(fname) % loads 'labels'
    labeling = graph_image_labeling(test_spIm{TestImgNum},labels,false);
    labeling = uint8(imresize(labeling,size(labeling)*2,'nearest'));
    
    if useDisplay
        imagesc(labeling);
        pause(0.1);
    end
    
    % save results 
    imName = test_imageNames{TestImgNum};
    imName = strsplit(imName,{'.'});
    imName = imName{1} + '.png';
    fname = strcat(DatasetHomeDir, SubmissionSubDir,imName);
    imwrite(labeling, fname);    
end

%-------------------------------


display('Done');
display('===========================');

end
