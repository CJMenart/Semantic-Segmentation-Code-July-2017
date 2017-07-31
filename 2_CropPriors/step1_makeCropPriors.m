function [] = step1_makeCropPriors(DatasetHomeDir, ModelSubDir, Sites)

% directories
ProbSubDir = strcat(ModelSubDir, 'Prob/');
TrainDir = strcat(DatasetHomeDir, 'TrainImgs/');
TestDir = strcat(DatasetHomeDir, 'TestImgs/');

TrainPriorDir = strcat(ModelSubDir, 'TrainPrior/');
% make sub directory for the files
if(~exist([DatasetHomeDir TrainPriorDir],'dir'))
    if(~mkdir(DatasetHomeDir, TrainPriorDir))
        fprintf(1,'Error creating %s... quitting\n', ...
                        [DatasetHomeDir TrainPriorDir]);
        return;
    end
end

% load metadata
fname = strcat(DatasetHomeDir,'metaData.mat');
load(fname); % loads 'metaData'
numLabels = metaData.numClasses;

% load classifier confusion: P( C | L )
fname = strcat(DatasetHomeDir, ModelSubDir, sprintf('Prob_CgivenL_%04d.mat', Sites));
load(fname); % loads  'Prob_CgivenL', etc.
     
for TT=1:2 % train, test
    
    if(TT==1)
        % use training images
        display('Loading training truth');
        fname = strcat(DatasetHomeDir, 'train_pixeltruth.mat');
        load(fname); % loads 'train_pixeltruth'
        truthAll = train_pixeltruth;
        clear train_pixeltruth;
    else
        % use testing images
        display('Loading testing truth');
        fname = strcat(DatasetHomeDir, 'test_pixeltruth.mat');
        load(fname); % loads 'test_pixeltruth'
        truthAll = test_pixeltruth;
        clear test_pixeltruth;
    end

    % num to use
    numImgs = length(truthAll);


    % process images
    for ImgNum = 1:numImgs

        fprintf(1,'\t******************\n');
        if(TT==1)
            fprintf(1,'\tTrain Image #%d\n', ImgNum);
            fname = strcat(TrainDir, sprintf('train_img_%06d.mat', ImgNum));
        else
            fprintf(1,'\tTest Image #%d\n', ImgNum);
            fname = strcat(TestDir, sprintf('test_img_%06d.mat', ImgNum));
        end
        % load rgbIm
        load(fname); % loads 'rgbIm'

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % IN
        %

        % load softmax probability inputs (for each (super)pixel)
        if(TT==1)
            fname = strcat(DatasetHomeDir, ProbSubDir, sprintf('train_%06d_prob_%04d.mat',ImgNum,Sites));
        else
            fname = strcat(DatasetHomeDir, ProbSubDir, sprintf('test_%06d_prob_%04d.mat',ImgNum,Sites));
        end
        load(fname); % loads 'Prob'
        In = double(Prob);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % OUT
        %

        % get target outputs
        truth = truthAll{ImgNum}(:);    

        % make one-hot outputs
        Out = zeros(size(In));
        for i=1:size(Out,2)
            if(truth(i)>0)
                Out(truth(i),i)=1;
            end
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        vggInputSize = 224;
        featureMapSize = 7;
        [cropCoords] = getImageCropWindows(rgbIm, vggInputSize);


        % initialize P_L
        P_L = zeros(numLabels,size(cropCoords,2));

        % solve for P_L for crops
        for C = 1:size(cropCoords,2)


            % use this crop (for now)
            useMe = 1;

            % get box
            crop_rows = cropCoords(1,C):cropCoords(3,C);
            crop_cols = cropCoords(2,C):cropCoords(4,C);

            % convert to index
            k=0;
            cropInd=[];
            for c=1:length(crop_cols)
                for r=1:length(crop_rows)
                    % convert to index
                    k=k+1;
                    cropInd(k)=sub2ind([size(rgbIm,1) size(rgbIm,2)], crop_rows(r),crop_cols(c));
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            truth_crop = truth(cropInd);
            In_crop = In(:,cropInd);
            Out_crop = Out(:,cropInd);


            % check to make sure not too many background 0 sites
            numZ = length(find(truth_crop==0));
            ratio = numZ/length(truth_crop);
            if((ratio > .1) && (TT==1)) % too many for training
                useMe = 0;
                fprintf(1,'\t\t***Too many 0 sites (%0.3f)\n', ratio);
            end

            subplot(1,4,1);
            imshow(rgbIm);
            if(TT==1)
                title(sprintf('Train Image #%d', ImgNum));
            else
                title(sprintf('Test Image #%d', ImgNum));
            end
            
            subplot(1,4,2);
            dispIm = zeros(size(rgbIm));
            dispIm(crop_rows,crop_cols,:) = rgbIm(crop_rows,crop_cols,:);
            imshow(dispIm/255);
            axis('image');
            title('Crop');

            subplot(1,4,3);
            dispIm = zeros(size(rgbIm,1),size(rgbIm,2));
            dispIm(crop_rows,crop_cols) = truthAll{ImgNum}(crop_rows,crop_cols);
            imagesc(dispIm,[0 numLabels]);
            axis('image');
            title('Truth');


            % check to see if have valid sites and ok to use  
            if(useMe) 

                % use only non-zero classes
                ind = find(truth_crop>0);
                truth_crop = truth_crop(ind);
                In_crop = In_crop(:,ind);
                Out_crop = Out_crop(:,ind);
                numSites = length(ind);

                % which are active truth labels
                validTruthLabels_crop = unique(truth_crop)';

                % initial guess of non-zero portion    
                P_L_lim = 0.5*ones(length(validTruthLabels_crop),1);

                % use fmincon with active-set method
                options = optimoptions('fmincon','Display', 'none', 'MaxIterations', 2000, 'Algorithm','active-set');

                % lower and upper bounds
                lb = zeros(size(P_L_lim)); % 0
                ub = ones(size(P_L_lim)); % 1

                % solve it
                P_L_lim = fmincon(@mycostfun, P_L_lim, [],[], [],[], lb, ub, [],options, ...
                                In_crop, Out_crop, Prob_CgivenL, validTruthLabels_crop);

                % sum to 1
                s = sum(P_L_lim);
                if(s>0)
                    P_L_lim = P_L_lim/s;
                end

                % keep these for non-zero classes
                P_L(validTruthLabels_crop, C)=P_L_lim;  

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                % create mapping
                MappingMatrix = zeros(numLabels, numLabels);
                % set probabilities
                for i=1:numLabels
                    for j=1:numLabels
                        MappingMatrix(i,j) = Prob_CgivenL(j,i)*P_L(i,C);
                    end
                end
                % normalize columns
                for j=1:numLabels
                    v = MappingMatrix(:,j);
                    sv=sum(v);
                    if(sv>0)
                        v=v/sv;
                    end
                    MappingMatrix(:,j)=v;
                end 

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                % truth labels
                truthLabels = vec2ind(Out_crop); 

                % orig labels
                origLabels = vec2ind(In_crop);

                % mapped labels
                mappedIn = MappingMatrix*In_crop;
                mapLabels = vec2ind(mappedIn);

                % acc
                if(numSites>0)
                    origAcc(C, ImgNum) = length(find(origLabels==truthLabels))/length(truthLabels);
                    mapAcc(C, ImgNum) = length(find(mapLabels==truthLabels))/length(truthLabels);
                end

                % make sure not do worse
                if(mapAcc(C,ImgNum) < origAcc(C,ImgNum))
                    fprintf(1,'\t\t(%.1f%%) origAcc = %.4f,  mapAcc = %.4f *** WORSE\n', 100*ratio, origAcc(C, ImgNum), mapAcc(C, ImgNum));
                else
                    fprintf(1,'\t\t(%.1f%%) origAcc = %.4f,  mapAcc = %.4f\n', 100*ratio, origAcc(C, ImgNum), mapAcc(C, ImgNum));
                end

            end



            subplot(1,4,4);
            bar(P_L(:,C));
            axis([1 numLabels 0 1]);
            axis('square');
            title('P(L)');

            pause(0.00001); 

        end % C

        % save
        if(TT==1)
            fname = strcat(DatasetHomeDir, TrainPriorDir, sprintf('train_%06d_trainprior_%04d.mat', ImgNum, Sites));
        else
            fname = strcat(DatasetHomeDir, TrainPriorDir, sprintf('test_%06d_trainprior_%04d.mat', ImgNum, Sites));
        end
        fprintf(1,'\t\tSaving %s\n', fname);
        probIm = reshape(Prob', [size(rgbIm,1), size(rgbIm,2), numLabels]);
        probIm = single(probIm);
        Prob_L = P_L;
        save(fname, 'rgbIm', 'probIm', 'cropCoords', 'Prob_L', '-v7.3');


    end % ImgNum

end



display('FINISHED!');


end
