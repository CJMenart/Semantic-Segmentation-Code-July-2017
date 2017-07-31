function [] = step1_getL2SectHypercolumns(DatasetHomeDir, ModelSubDir, numSP, VGGfname, whichVGGHCLayersToUse)

% storage
HCSubDir =  'HC/';
% make sub directory for the hypercolumn files
if(~exist([DatasetHomeDir HCSubDir],'dir'))
    if(~mkdir(DatasetHomeDir, HCSubDir))
        fprintf(1,'Error creating %s... quitting\n', [DatasetHomeDir HCSubDir]);
        pause;
        return;
    end
end


TrainDir = strcat(DatasetHomeDir, 'TrainImgs/');
TestDir = strcat(DatasetHomeDir, 'TestImgs/');

% setup nn path
display('Loading VGG');
run vl_setupnn
VGG = load(VGGfname);
VGG = vl_simplenn_tidy(VGG); % need to clean up (adds dilate field)


for TT=1:2 % train, test
    
    if(TT==1)
        display('Loading training superpixels');
        fname = strcat(DatasetHomeDir, sprintf('train_spIm_%04d.mat', numSP));
        load(fname); % loads 'train_spIm'
        TT_spIm = train_spIm;
        clear train_spIm;
    else
        display('Loading testing superpixels');
        fname = strcat(DatasetHomeDir, sprintf('test_spIm_%04d.mat', numSP));
        load(fname); % loads 'test_spIm'
        TT_spIm = test_spIm;
        clear test_spIm;
    end
    
    display('Processing images');
    numImgs = length(TT_spIm);
    for i=1:numImgs

        % display
        if(TT==1)
            fprintf(1,'Train #%d\n',i); 
        else
        	fprintf(1,'Test #%d\n',i); 
        end
            
        % load image
        if(TT==1)
            fname = strcat(TrainDir, sprintf('train_img_%06d.mat', i));
        else
            fname = strcat(TestDir, sprintf('test_img_%06d.mat', i));
        end
        load(fname); % loads 'rgbIm'

        % get superpixels
        spIm = TT_spIm{i};

        % pass full image through VGG
        res = getFullVGG(VGG, rgbIm);

        % get hypercolumns for each superpixel center (with L2-norm of each layer part)
        spHC = getVGGL2SectHypercolumn(spIm, res, whichVGGHCLayersToUse);

        % save superpixel hypercolumns for image
        if(TT==1)
            fname = strcat(DatasetHomeDir, HCSubDir, sprintf('train_%06d_hc_%04d.mat', i, numSP));
        else
            fname = strcat(DatasetHomeDir, HCSubDir, sprintf('test_%06d_hc_%04d.mat', i, numSP));
        end
        save(fname, 'spHC', '-v7.3');

        % need to clear!!!
        clear res;

    end

    display('FINISHED!');

end


clear VGG;


end