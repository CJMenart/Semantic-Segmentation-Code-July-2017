function [] = step1_sameDiffClassification(DatasetHomeDir,ModelSubDir,settings)
% Trains a Siamese Neural Network on a distance metric for image patches.
% Dependencies: Tensorflow 1.2, Python 3.5

display('Loading metadata');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads metaData
numTraining = metaData.numTrain;

%make a bunch of folders
mkdir_safe(settings.sameDiffDir);
mkdir_safe(settings.trainExampleDir);
mkdir_safe(settings.trainDataDir);
mkdir_safe(settings.checkpointDir);
mkdir_safe(settings.testDataDir);
    
% In order to call our python scripts, we must add the folder to PYTHONPATH
pathToScripts = strcat(settings.codeDir,'6_SameDiffClassification/');
pythonpath = py.sys.path;
if count(pythonpath,pathToScripts) == 0
    insert(pythonpath,int32(0),pathToScripts);
end

% create a small file to just keep track of what we've already done, in
% case we crash halfway through (this step is rather long). Some steps will
% simply start over, but the training and csv writing, the longest steps,
% should be able to continue where they left off.
pfl = strcat(settings.sameDiffDir,'progress.mat');
if ~exist(pfl,'file')
    progress = [];
    progress.isTraindatMade = false;
    progress.isTraindatWritten = false;
    progress.isTrainingDone = false;
    progress.isTestdatMade = false;
    progress.isTestingDone = false;
else
    load(pfl);
end

% make training data
if ~progress.isTraindatMade
    folders = dir(settings.trainExampleDir);
    folders = folders(3:end); %remove '.', '..'. Works bc all subdir start with letters
    if ~isempty(folders)
        rmdir(folders.name,'s');
    end
    disp('CHOOSING TRAINING DATA')
    if settings.directToText
        create_samediff_traindata_direct_to_text(DatasetHomeDir,ModelSubDir,settings);
    else
        ds = create_samediff_traindata(DatasetHomeDir,ModelSubDir,settings);
    end
    progress.isTraindatMade = true;
    save(pfl,'progress');
end
if ~progress.isTraindatWritten && ~settings.directToText
    if ~exist('ds','var')
        load([settings.sameDiffDir 'TrainingExamples/trainingExampleDatastore']);
    end
    disp('WRITING TRAINING DATA IN TEXT BATCHES')
    write_samediff_traindata_to_text(ds,settings);
    progress.isTraindatWritten = true;
    save(pfl,'progress');
end
    
%train neural network
if ~progress.isTrainingDone
    disp('TRAINING NETWORK')
    py.sameDiffClassification.sameDiffTraining(settings.checkpointDir,settings.trainDataDir,...
        'samediff_traindat','samediff_valdat',numTraining*settings.examplesPerImage);
    progress.isTrainingDone = true;
    save(pfl,'progress');
end
    
% make testing data
if ~progress.isTestdatMade
    disp('CHOOSING TESTING DATA')
    create_samediff_testdata(DatasetHomeDir,ModelSubDir,settings);
    progress.isTestdatMade = true;
    save(pfl,'progress');
end

% classify test data
if ~progress.isTestingDone
    disp('RUNNING TRAINED NETWORK')
    py.sameDiffClassification.sameDiffTesting(settings.checkpointDir,settings.testDataDir,'samediff_testdat');
    progress.isTestingDone = true;
    save(pfl,'progress');
end

%optional: display the connections
if settings.show
    display_samediff_testdata(DatasetHomeDir,ModelSubDir,settings);
end
    
end