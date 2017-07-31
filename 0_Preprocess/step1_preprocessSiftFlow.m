function [] = step1_preprocessSIFTFLow(InputRepositoryDir, DatasetHomeDir)


% where to read images and truth
ImagesSubDir = 'Images/spatial_envelope_256x256_static_8outdoorcategories/';
TruthSubDir = 'SemanticLabels/spatial_envelope_256x256_static_8outdoorcategories/';

TrainDir = strcat(DatasetHomeDir, 'TrainImgs/');
% make sub directory for the files
if(~exist(TrainDir,'dir'))
    if(~mkdir(TrainDir))
        fprintf(1,'Error creating %s... quitting\n', ...
                        TrainDir);
        return;
    end
end

TestDir = strcat(DatasetHomeDir, 'TestImgs/');
% make sub directory for the files
if(~exist(TestDir,'dir'))
    if(~mkdir(TestDir))
        fprintf(1,'Error creating %s... quitting\n', ...
                        TestDir);
        return;
    end
end


% get filenames of rgb images
imageD=dir(strcat(InputRepositoryDir, ImagesSubDir, '*.jpg'));

% get filenames of truth images
truthD=dir(strcat(InputRepositoryDir, TruthSubDir, '*.mat'));

% get testing filenames
fname = strcat(InputRepositoryDir,'TestSet1.txt');
testNames = importdata(fname, '\n');

% process images
display('Processing images');
numTrain=0;
numTest=0;
for i=1:length(imageD)
    
    % filename
    imageName = imageD(i).name;
    truthName = truthD(i).name;
    %display(sprintf('Image %d: %s',i, imageName));
    
   
    % full path filename
    imageFullfname = strcat(InputRepositoryDir, ImagesSubDir, imageName);
    truthFullfname = strcat(InputRepositoryDir, TruthSubDir, truthName);

    % load image
    rgbIm = imread(imageFullfname);
    
    % load truth
    truth = load(truthFullfname);
    truthIm = truth.S;
    
    % save labels
    if(i==1)
        classLabels = truth.names;
    end
    
    
    % check if test data
    memInd = checkMembershipTestNames(strcat(['spatial_envelope_256x256_static_8outdoorcategories\' imageName]), testNames);
    if(memInd>0) % test image
        numTest = numTest+1;
         
        %test_imgs{numTest} = rgbIm;
        fname = strcat(TestDir, sprintf('test_img_%06d.mat', numTest));
        save(fname, 'rgbIm', '-v7.3');

        test_pixeltruth{numTest} = truthIm;
    	metaData.test_imageNames{numTest} = imageName;
    else % train image
        numTrain = numTrain+1; 
        
        %train_imgs{numTrain} = rgbIm;       
        fname = strcat(TrainDir, sprintf('train_img_%06d.mat', numTrain));
        save(fname, 'rgbIm', '-v7.3');
        
        train_pixeltruth{numTrain} = truthIm;
    	metaData.train_imageNames{numTrain} = imageName;
    end
        
end



% create/save metadata
metaData.numTrain = numTrain;
metaData.numTest = numTest;
metaData.whichTrainAreVal = [];
metaData.classLabels = classLabels;
metaData.numClasses = length(classLabels);
fname = strcat(DatasetHomeDir,'metaData.mat');
save(fname, 'metaData', '-v7.3');


display('Saving training data');
%fname = strcat(DatasetHomeDir,'train_imgs.mat');
%save(fname, 'train_imgs', '-v7.3');
fname = strcat(DatasetHomeDir,'train_pixeltruth.mat');
save(fname, 'train_pixeltruth', '-v7.3');
display('Done');


display('Saving testing data');
%fname = strcat(DatasetHomeDir,'test_imgs.mat');
%save(fname, 'test_imgs', '-v7.3');
fname = strcat(DatasetHomeDir,'test_pixeltruth.mat');
save(fname, 'test_pixeltruth', '-v7.3');



end
