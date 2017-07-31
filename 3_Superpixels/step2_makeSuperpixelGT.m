function [] = step2_makeSuperpixelGT(DatasetHomeDir, numSP)
      
display('MAKE SUPERPIXEL GT');

% load metadata
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'
numClasses = metaData.numClasses;


for TT=1:2 % train, test
    
    if(TT==1)
        
        % train pixel truth
        display('Loading training data');
        fname = strcat(DatasetHomeDir, 'train_pixeltruth.mat');
        load(fname); % loads 'train_pixeltruth'
        TT_pixeltruth = train_pixeltruth;
        clear train_pixeltruth;

        % train superpixels
        fname = strcat(DatasetHomeDir, sprintf('train_spIm_%04d.mat',numSP));
        load(fname); % loads 'train_spIm'
        TT_spIm = train_spIm;
        clear train_spIm;
        
    else
        
        % test pixel truth
        display('Loading testing data');
        fname = strcat(DatasetHomeDir, 'test_pixeltruth.mat');
        load(fname); % loads 'test_pixeltruth'
        TT_pixeltruth = test_pixeltruth;
        clear test_pixeltruth;

        % test superpixels
        fname = strcat(DatasetHomeDir, sprintf('test_spIm_%04d.mat',numSP));
        load(fname); % loads 'test_spIm'
        TT_spIm = test_spIm;
        clear test_spIm;
        
    end

    % process images
    numImgs=length(TT_spIm);
    for i=1:numImgs

%         if(TT==1)
%             fprintf(1, 'Train Image %d\n',i);
%         else
%             fprintf(1, 'Test Image %d\n',i);
%         end
        
        % get truth image
        gtIm = TT_pixeltruth{i};

        % get superpixel image
        spIm = TT_spIm{i};

        % get/save superpixel ground truth
        spGT = assignGTtoSP(gtIm, spIm);
        if(TT==1)
            train_spGT{i} = spGT;
        else
            test_spGT{i} = spGT;
        end

        % show it
        subplot(1,2,1);
        imagesc(gtIm,[0 numClasses]);
        axis('image');
        colormap('jet');
        if(TT==1)
            title(sprintf('Truth Train Image %d', i));
        else
            title(sprintf('Truth Test Image %d', i));
        end
        subplot(1,2,2);
        dispIm=zeros(size(gtIm));
        for j = 1:length(spGT)
            ind=find(spIm==j);
            dispIm(ind)=spGT(j);
        end
        imagesc(dispIm, [0 numClasses]);
        axis('image');
        colormap('jet');
        title('GT Superpixels');
        pause(0.001);


    end

    display('Saving');
    if(TT==1)
        fname = strcat(DatasetHomeDir, sprintf('train_spGT_%04d.mat', numSP));
        save(fname, 'train_spGT', '-v7.3');
    else
        fname = strcat(DatasetHomeDir, sprintf('test_spGT_%04d.mat', numSP));
        save(fname, 'test_spGT', '-v7.3');
    end
    
end


end