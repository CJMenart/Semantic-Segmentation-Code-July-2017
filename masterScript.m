
% Master script
clear all; close all;


%--------------------------------------
%
% INPUT SETTINGS
%

% where to read original data
%----------------------------
InputRepositoryDir = 'C:\Users\Christopher\Documents\Computer Vision Overflow\SIFT-Flow\';

% where to store processed data
%------------------------------
DatasetHomeDir = 'C:\Users\Christopher\Documents\Computer Vision Overflow\SIFT-Flow\';

% add path for matconvnet
%------------------------------
MatconvnetDir = 'C:\Users\Christopher\Documents\Computer Vision Overflow\SIFT-Flow\';
addpath(MatconvnetDir);


% make sure to call the correct dataset preprocessing script below!
%------------------------------------------------------------------
%XXXXXXXX see step-0 below! XXXXXXXX


% storage for this model 
%-----------------------
ModelSubDir = 'Model_Long/';
Sites = [0]; % Long - no superpixels

%--------------------------------------

% target number of superpixels
desiredNumSP = 500;
spcompactness = 10;

%--------------------------------------

% VGG net
VGGfname = sprintf('%sModels/imagenet-vgg-verydeep-16.mat', MatconvnetDir);
    
% which HC layers to use
whichVGGHCLayersToUse = [10 17 31]; % [pool2, pool3, pool5] [128+256+512]=896 features

%------------------------------------

%Stage 6 Settings
sameDiffSettings.examplesPerImage = 250;
sameDiffSettings.examplesPerFile = 1000;
sameDiffSettings.neighborhoodRadius = 30;
sameDiffSettings.codeDir = 'C:\Users\Christopher\Documents\Computer Vision Overflow\Semantic Segmentation Code July 2017/';
sameDiffSettings.sameDiffDir = strcat(DatasetHomeDir,ModelSubDir,sprintf('SameDiffClassification_%04d/',desiredNumSP));
sameDiffSettings.trainExampleDir = strcat(sameDiffSettings.sameDiffDir,'TrainingExamples/');
sameDiffSettings.trainDataDir = strcat(sameDiffSettings.sameDiffDir,'Training Examples in CSV Format/');
sameDiffSettings.checkpointDir = strcat(sameDiffSettings.sameDiffDir,'Network Checkpoints/');
sameDiffSettings.testDataDir = strcat(sameDiffSettings.sameDiffDir,'Test Image Pairs/');
sameDiffSettings.show = true;
sameDiffSettings.valProp = 0.05;
sameDiffSettings.spVersion = desiredNumSP;
sameDiffSettings.split = 'train';
sameDiffSettings.directToText = true;
%----------------------------------

% Stage 7 settings
lambda = 0.2;

% --------------------------------
DO_STAGES = [ 6:8 ]; % must do in order the first time! Then do (6),(7) for outside probabilities

%--------------------------------------
%--------------------------------------






%--------------------------------------
%
% validate directories
%

% check directory
if(~exist(InputRepositoryDir,'dir'))
	fprintf(1,'Error findind %s... quitting\n', InputRepositoryDir);
	return;
end

% make directory if needed
if(~exist(DatasetHomeDir,'dir'))
    if(~mkdir(DatasetHomeDir))
        fprintf(1,'Error creating %s... quitting\n', DatasetHomeDir);
        return;
    end
end

% make sub directory for the files
if(~exist([DatasetHomeDir ModelSubDir],'dir'))
    if(~mkdir(DatasetHomeDir, ModelSubDir))
        fprintf(1,'Error creating %s... quitting\n', ...
                        [DatasetHomeDir ModelSubDir]);
        return;
    end
end

%--------------------------------------



%--------------------------------------
%
% preprocess dataset
%

if(ismember(0, DO_STAGES))
    
    display('********* PREPROCESS *********');
    
    addpath('0_Preprocess/');

    % reformat data
    step1_preprocessSiftFlow(InputRepositoryDir, DatasetHomeDir);

    % examine rgb and truth images
    test_displayTruth(DatasetHomeDir); 
    
    display('********* DONE *********');

end

%--------------------------------------



%--------------------------------------
%
% build confusion matrices
%

if(ismember(1, DO_STAGES))
    
    display('********* CONFUSION MATRICES *********');

    display('NOTE:  DO THIS FOR VALIDATION DATA ONLY!!!');

    addpath('1_Confusion/');

    % get probability confusion mapping P(L|C)
    step1_getProbCgivenL(DatasetHomeDir, ModelSubDir, Sites);

    display('********* DONE *********');

end


%--------------------------------------


%--------------------------------------
%
% build confusion priors
%

if(ismember(2, DO_STAGES))
    
    display('********* CONFUSION PRIORS *********');

    display('NOTE:  DO THIS FOR VALIDATION DATA ONLY!!!');
    
    addpath('2_CropPriors/');

    step1_makeCropPriors(DatasetHomeDir, ModelSubDir, Sites);
    
    display('********* DONE *********');

end


%--------------------------------------


%--------------------------------------
%
% build superpixels 
%

if(ismember(3, DO_STAGES))
    
    display('********* SUPERPIXELS *********');
    
    addpath('3_Superpixels/');

    step1_makeSuperpixels(DatasetHomeDir, desiredNumSP, spcompactness);
    
    step2_makeSuperpixelGT(DatasetHomeDir, desiredNumSP);
    
    step3_makeSuperpixelMaxProb(DatasetHomeDir, ModelSubDir, Sites, desiredNumSP)
    
    display('********* DONE *********');

end


%--------------------------------------



%--------------------------------------
%
% build hypercolumns 
%

if(ismember(4, DO_STAGES))
    
    display('********* HYPERCOLUMNS *********');
    
    addpath('4_Hypercolumns/');

    step1_getL2SectHypercolumns(DatasetHomeDir, ModelSubDir, desiredNumSP, VGGfname, whichVGGHCLayersToUse);

    display('********* DONE *********');

end


%--------------------------------------

if(ismember(6,DO_STAGES))

    display('********** SAME-DIFFERENT CLASSIFICATION ************');
    step1_sameDiffClassification(DatasetHomeDir,ModelSubDir,sameDiffSettings);
end

%---------------------------------------

if(ismember(7,DO_STAGES))
   display('*********** CRF LABELING **************');
   step0_reshapeMappedProbs(DatasetHomeDir,ModelSubDir,desiredNumSP);
   step1_CRFLabels(DatasetHomeDir,ModelSubDir,desiredNumSP,lambda,sameDiffSettings.neighborhoodRadius);
end

%---------------------------------------

if (ismember(8,DO_STAGES))
    step1_evaluateResults(DatasetHomeDir, ModelSubDir, desiredNumSP, true);
end
%----------------------------------------

display('FINISHED ALL!');

