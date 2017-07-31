%convert


DatasetHomeDir = '/Users/jwdavis/Desktop/Data5/ProcessedData/SIFT-Flow/';

ModelSubDir = 'Model_Long/';
ProbSubDir = strcat(ModelSubDir, 'Prob/');



%
% TRAINING
%

% % RENAME
% for i=1:2488
%     str1 = sprintf('%s%sTrain_%03d_softmax65535.mat', DatasetHomeDir, ProbSubDir, i);
%     str2 = sprintf('%s%strain_%04d_prob_0000.mat', DatasetHomeDir, ProbSubDir, i);
%     cmd = sprintf('mv %s %s', str1, str2);
%     system(cmd);
% end
% 
% % CONVERT VARIABLE
% for i=1:2488
%     str = sprintf('%s%strain_%04d_prob_0000.mat', DatasetHomeDir, ProbSubDir, i);
%     load(str); % loads SoftMax
%     Prob = single(SoftMax);
%     save(str,'Prob', '-v7.3');
% end




%
% TESTING
%

% % RENAME
% for i=1:2488
%     str1 = sprintf('%s%sTest_%03d_softmax65535.mat', DatasetHomeDir, ProbSubDir, i);
%     str2 = sprintf('%s%stest_%04d_prob_0000.mat', DatasetHomeDir, ProbSubDir, i);
%     cmd = sprintf('mv %s %s', str1, str2);
%     system(cmd);
% end

% % CONVERT VARIABLE
% for i=1:200
%     str = sprintf('%s%stest_%04d_prob_0000.mat', DatasetHomeDir, ProbSubDir, i);
%     load(str); % loads SoftMax
%     Prob = single(SoftMax);
%     save(str,'Prob', '-v7.3');
% end


