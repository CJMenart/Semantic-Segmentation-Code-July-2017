function [] = displayTruth(DatasetHomeDir) 

display('DISPLAY TRUTH');

TrainDir = strcat(DatasetHomeDir, 'TrainImgs/');
TestDir = strcat(DatasetHomeDir, 'TestImgs/');

                
                
% load metadata
display('Loading metaData');
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'


for TT=1:2 % 1:train, 2:test

	if(TT==1)
        display('Loading training data');

		fname = strcat(DatasetHomeDir,'train_pixeltruth.mat');
		load(fname); % loads 'train_pixeltruth'
		tt_pixeltruth = train_pixeltruth;
		clear train_pixeltruth;
    else
        display('Loading testing data');

		fname = strcat(DatasetHomeDir,'test_pixeltruth.mat');
		load(fname); % loads 'test_pixeltruth'
		tt_pixeltruth = test_pixeltruth;
		clear test_pixeltruth;
	end


	% processing
    numImgs=length(tt_pixeltruth);
	for i=1:numImgs
  
    		% display it
    		subplot(1,2,1);
            if(TT==1)
                fname = strcat(TrainDir, sprintf('train_img_%06d.mat', i));
                load(fname); % loads 'rgbIm'
                imshow(rgbIm); axis('image');
    			title(sprintf('Train %d',i));
            else
                title(sprintf('Test %d',i));
                fname = strcat(TestDir, sprintf('test_img_%06d.mat', i));
                load(fname); % loads 'rgbIm'
                imshow(rgbIm); axis('image');
    			title(sprintf('Test %d',i));
            end
                        
            subplot(1,2,2);
    		imagesc(tt_pixeltruth{i},[0 metaData.numClasses]); axis('image');
    		title('Truth');
            
    		pause(0.1);

	end


end

close

end