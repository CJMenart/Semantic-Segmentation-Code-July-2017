function [spProb,spMaxProb] = assignProbtoSP(Prob, spIm)

% number of superpixels
numSP = max(max(spIm));
numClasses = size(Prob,1);

% save max class for each superpixel
spMaxProb = zeros(numSP,1);
spProb = zeros(numClasses,numSP);

% assign truth
for sp = 1:numSP
    
	% find the pixels belonging to superpixel
	ind = find(spIm==sp);	
   
	% save mean
	vec = mean(Prob(:,ind),2);
    spProb(:,sp) = vec;
    spMaxProb(sp) = vec2ind(vec);

end


end