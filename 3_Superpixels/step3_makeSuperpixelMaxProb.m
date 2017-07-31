function [] = step3_makeSuperpixelMaxProb(DatasetHomeDir, ModelSubDir, Sites, numSP)
      
display('MAKE SUPERPIXEL PROB AND SUPERPIXEL MAX PROB');

% sub directories
ProbSubDir = strcat(ModelSubDir, 'Prob/');

% load metadata
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'
numClasses = metaData.numClasses;


for TT=1:2 % train, test
    
    if(TT==1)
        % training superpixels
        fname = strcat(DatasetHomeDir, sprintf('train_spIm_%04d.mat',numSP));
        load(fname); % loads 'train_spIm'
        TT_spIm = train_spIm;
        clear train_spIm;
        
        fname = strcat(DatasetHomeDir, sprintf('train_spGT_%04d.mat',numSP));
        load(fname); % loads 'train_spGT'
        TT_spGT = train_spGT;
        clear train_spGT;
    else
        % testing superpixels
        fname = strcat(DatasetHomeDir, sprintf('test_spIm_%04d.mat',numSP));
        load(fname); % loads 'test_spIm'
        TT_spIm = test_spIm;
        clear test_spIm; 
        
        fname = strcat(DatasetHomeDir, sprintf('test_spGT_%04d.mat',numSP));
        load(fname); % loads 'test_spGT'
        TT_spGT = test_spGT;
        clear test_spGT; 
    end
        
    % process images
    numImgs=length(TT_spIm);
    for i=1:numImgs

%         if(TT==1)
%             fprintf(1, 'Train Image %d\n',i);
%         else
%             fprintf(1, 'Test Image %d\n',i);
%         end


        % get superpixel image
        spIm = TT_spIm{i};

        % load softmax probability 
        clear Prob;
        if(TT==1)
            fname = strcat(DatasetHomeDir, ProbSubDir, sprintf('train_%06d_prob_%04d.mat',i, Sites));
        else
            fname = strcat(DatasetHomeDir, ProbSubDir, sprintf('test_%06d_prob_%04d.mat',i, Sites));
        end
        load(fname); % loads 'Prob'

        % get/save superpixel max probability
        [Prob, spMaxProb] = assignProbtoSP(Prob, spIm);

        if(TT==1)
            train_spMaxProb{i} = spMaxProb;
        else
            % NOTE: In the final pipeline we should not actually need this
            % prob. This was added by Chris to allow him to test CRF code
            % before the Prior Estimation part of the code was integrated.
            % It can be removed later.
            fname = strcat(DatasetHomeDir, ProbSubDir, sprintf('test_%06d_prob_%04d',i,numSP));
            save(fname,'Prob');
            test_spMaxProb{i} = spMaxProb;
        end
        
        
        subplot(1,2,1);
        dispIm=zeros(size(spIm));
        for j = 1:length(TT_spGT{i})
            ind=find(spIm==j);
            dispIm(ind)=TT_spGT{i}(j);
        end
        imagesc(dispIm, [0 numClasses]);
        axis('image');
        colormap('jet');
        if(TT==1)
            title(sprintf('Train spGT %d', i));
        else
            title(sprintf('Test spGT %d', i));
        end
        
        subplot(1,2,2);
        for j = 1:length(TT_spGT{i})
            ind=find(spIm==j);
            dispIm(ind)=spMaxProb(j);
        end
        imagesc(dispIm, [0 numClasses]);
        axis('image');
        colormap('jet');
        title('MaxProb Superpixels');
        
        pause(0.001);   

    end
    
    display('Saving');
    if(TT==1)
        fname = strcat(DatasetHomeDir, ModelSubDir, sprintf('train_spMaxProb_%04d.mat', numSP));
        save(fname, 'train_spMaxProb', '-v7.3');
    else
        fname = strcat(DatasetHomeDir, ModelSubDir, sprintf('test_spMaxProb_%04d.mat', numSP));
        save(fname, 'test_spMaxProb', '-v7.3');
    end

end

end

