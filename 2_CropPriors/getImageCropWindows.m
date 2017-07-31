
function [ cropCoords ] = getImageCropWindows(Im, netInputSize)
% [ cropCoords ] = getImageCropWindows(Im, netInputSize)
%
% Im = input RGB image.
%
% netInputSize =  default width of the *square* image input to the net.
%
% cropCoords = matrix of upper-left and bottom-right row,col coordinates
% (one box per column).
%



% enlarge image size if needed
useOnlyOneCrop = 0;
if(min(size(Im,1), size(Im,2))<netInputSize)
    if(size(Im,1)<=size(Im,2))
        Im = imresize(Im, [netInputSize NaN]);
    else
        Im = imresize(Im, [NaN netInputSize]);
    end
    useOnlyOneCrop=1;
end

%% UPPER-LEFT
crop=1;
% upper-left row,col coord
cropCoords(1,crop) = 1;
cropCoords(2,crop) = 1;
% lower-right row, col coord
cropCoords(3,crop) = cropCoords(1,crop) + netInputSize - 1;
cropCoords(4,crop) = cropCoords(2,crop) + netInputSize - 1;

if(useOnlyOneCrop)
    return;
end

%% UPPER-RIGHT
crop=crop+1;
% upper-left row,col coord
cropCoords(1,crop) = 1;
cropCoords(2,crop) = size(Im,2) - netInputSize + 1;
% lower-right row, col coord
cropCoords(3,crop) = cropCoords(1,crop) + netInputSize - 1;
cropCoords(4,crop) = cropCoords(2,crop) + netInputSize - 1;


%% LOWER-LEFT
crop=crop+1;
% upper-left row,col coord
cropCoords(1,crop) = size(Im,1) - netInputSize + 1;
cropCoords(2,crop) = 1;
% lower-right row, col coord
cropCoords(3,crop) = cropCoords(1,crop) + netInputSize - 1;
cropCoords(4,crop) = cropCoords(2,crop) + netInputSize - 1;


%% LOWER-RIGHT
crop=crop+1;
% upper-left row,col coord
cropCoords(1,crop) = size(Im,1) - netInputSize + 1;
cropCoords(2,crop) = size(Im,2) - netInputSize + 1;
% lower-right row, col coord
cropCoords(3,crop) = cropCoords(1,crop) + netInputSize - 1;
cropCoords(4,crop) = cropCoords(2,crop) + netInputSize - 1;


%% CENTER
crop=crop+1;
% upper-left row,col coord
cropCoords(1,crop) = floor(size(Im,1)/2) - floor(netInputSize/2) + 1;
cropCoords(2,crop) = floor(size(Im,2)/2) - floor(netInputSize/2) + 1;
% lower-right row, col coord
cropCoords(3,crop) = cropCoords(1,crop) + netInputSize - 1;
cropCoords(4,crop) = cropCoords(2,crop) + netInputSize - 1;


%%
        


end



