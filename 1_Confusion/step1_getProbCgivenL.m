function [] = step1_getProbCgivenL(DatasetHomeDir, ModelSubDir, Sites)

% sub directories
ProbSubDir = strcat(ModelSubDir, 'Prob/');

% load metaData
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'
numClasses = metaData.numClasses;
 

% load files
display('Loading train pixel truth');  
fname = strcat(DatasetHomeDir,'train_pixeltruth.mat');
load(fname); % loads 'train_pixeltruth'
numImgs = length(train_pixeltruth);
    



NumCgivenL = zeros(numClasses, numClasses);
for i=1:numImgs

    fprintf(1, 'Train Image %d\n',i);

        
    % each pixel is used
    GT = train_pixeltruth{i};
    GT = GT(:);


    % load probability per pixel
    fname = strcat(DatasetHomeDir, ProbSubDir, sprintf('train_%06d_prob_%04d.mat',i,Sites));
    load(fname); % loads Prob
    
    % get counts
    NumCgivenL = NumCgivenL + getNumCgivenL(GT, Prob);

end


% make sure full 
ind = find(NumCgivenL==0);
NumCgivenL(ind) = 0.001; % put in small value



% make into probabilities P(C|L)
Prob_CgivenL = zeros(numClasses, numClasses);
for i=1:numClasses
    v = NumCgivenL(:,i);
    sv = norm(v,1);
    if(sv>0)
        v=v/sv;
    end
    Prob_CgivenL(:,i) = v;
end


% show it
subplot(1,2,1);
imagesc(NumCgivenL);
axis('equal');
title('numCgivenL');

subplot(1,2,2);
imagesc(Prob_CgivenL);
axis('equal');
title('Prob(C|L)');
pause(0.01);

% save
fname = strcat(DatasetHomeDir, ModelSubDir, sprintf('Prob_CgivenL_%04d.mat',Sites));
save(fname, 'Prob_CgivenL', 'NumCgivenL', '-v7.3');


display('FINISHED!');


end