function [] = write_samediff_traindata_to_text(datastore,settings)
% take in the training data for the same-different classifier as a tall
% array, then shuffle and write out to a series of files so that you can
% read it in. In theory, this will in turn be read by tensorflow, which
% does a limited amount of shuffling by picking filenames in a random order
% and trying to mix together examples from a few files every time it
% iterates through.
% Dependencies: MATLAB 2017a (for tall-array functionality)

%load
tallTraindat = tall(datastore);
sz = size(tallTraindat);
sz = gather(sz);
numExamples = sz(1);
exampleLen = sz(2);
indices = 1:numExamples;

% get mean of training data and save so that you can mean-subtract both
% training and testing data
meanVec = mean(tallTraindat,1);
meanVec = gather(meanVec);
meanVec = meanVec(2:end);
meanVec = (meanVec(1:length(meanVec)/2)+meanVec(length(meanVec)/2+1:end))/2;
save([settings.sameDiffDir 'meanVec'],'meanVec');

%deterministic randomstream in case the process crashes halfway through...
rs = RandStream('mt19937ar','Seed',1492);

% pull out and save validation images
%TODO: Actually write this, and possibly move all your classification
%settings into a settings struct.
fnum = 0;
numVal = round(settings.valProp*numExamples/settings.examplesPerImage);
for ex = 1:numVal
    
    imNum = randi(rs, (numExamples/settings.examplesPerImage)-fnum);
    startInd = (imNum-1)*settings.examplesPerImage+1;
    endInd = startInd+settings.examplesPerImage-1;
    inds = indices(startInd:endInd);
    indices(startInd:endInd) = [];
    
    fnum=fnum+1;
    fname = [settings.trainDataDir sprintf('\\samediff_valdat_%d.csv',fnum)];
    if exist(fname,'file')
        fprintf('Found file %s, skipping...\n',fname);
        continue;
    end
    example = tallTraindat(inds,1:exampleLen);
    example = gather(example);
    example(:,2:end) = example(:,2:end) - horzcat(meanVec,meanVec);
    csvwrite(fname,example);
end

shuffling = indices(randperm(rs,length(indices)));
numExamples = length(shuffling);

fnum = 0;
for ex = 1:settings.examplesPerFile:numExamples
    fnum=fnum+1;
    fname = [settings.trainDataDir sprintf('\\samediff_traindat_%d.csv',fnum)];
    
    if exist(fname,'file')
        fprintf('Found file %s, skipping...\n',fname);
        continue;
    end
    
    inds = shuffling(ex:min(ex+settings.examplesPerFile-1, numExamples));
    inds = sort(inds);
    example = tallTraindat(inds,1:exampleLen);
    example = gather(example);
    example(:,2:end) = example(:,2:end) - horzcat(meanVec,meanVec);
    
    csvwrite(fname,example);
end
