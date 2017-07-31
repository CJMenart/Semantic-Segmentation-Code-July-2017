function [] = step1_makeSuperpixels(DatasetHomeDir, desiredNumSP, spcompactness)

display('GET SUPERPIXELS');

TrainDir = strcat(DatasetHomeDir, 'TrainImgs/');
TestDir = strcat(DatasetHomeDir, 'TestImgs/');

% load metadata
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'


% process testing images
% display('Loading training images');
% fname = strcat(DatasetHomeDir,'train_imgs.mat');
% load(fname); % loads 'train_imgs'

for TT = 1:2 % train, test

    if(TT==1)
        numImgs = metaData.numTrain;
    else
        numImgs = metaData.numTest;
    end
    
    for i=1:numImgs

        % load image
        if(TT==1)
            fname = strcat(TrainDir, sprintf('train_img_%06d.mat', i));
        else
            fname = strcat(TestDir, sprintf('test_img_%06d.mat', i));
        end
        load(fname); % loads 'rgbIm'

        % get superpixels
        [spIm, numSP]=superpixels(rgbIm, desiredNumSP, 'compactness', spcompactness);

        % save it
        if(TT==1)
            train_spIm{i} = spIm;
        else
            test_spIm{i} = spIm;
        end

        % show it
        subplot(1,2,1);
        imshow(rgbIm);
        if(TT==1)
            title(sprintf('Train Image %d', i));
        else
            title(sprintf('Test Image %d', i));
        end
        subplot(1,2,2);
        bmask = boundarymask(spIm);
        imshow(imoverlay(rgbIm*0.75,bmask,'black'));
        title('Superpixels');
        pause(0.001);

    end

    display('Saving');  
    if(TT==1)
        fname = strcat(DatasetHomeDir, sprintf('train_spIm_%04d.mat',desiredNumSP));
        save(fname, 'train_spIm', '-v7.3');
    else
        fname = strcat(DatasetHomeDir, sprintf('test_spIm_%04d.mat',desiredNumSP));
        save(fname, 'test_spIm', '-v7.3');
    end

end


end