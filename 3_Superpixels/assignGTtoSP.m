function [spGT] = assignGTtoSP(gtIm, spIm)

% number of superpixels
numSP = max(max(spIm));

% save ground truth for each superpixel
spGT = zeros(numSP,1);

% assign truth
for sp = 1:numSP
    
	% find the pixels belonging to superpixel
	ind = (spIm==sp);	

	% get the labels in the superpixel
	spLabels = gtIm(ind);

	% get mode (including 0 - don't care label)
	spGT(sp) = mode(spLabels);

end


end