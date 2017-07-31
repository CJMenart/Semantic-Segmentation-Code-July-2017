function [NumCgivenL]=getNumCGivenLabel(GT, Prob)


numClasses = size(Prob,1);
NumCgivenL = zeros(numClasses,numClasses);

% get NN best recommendation
C = vec2ind(Prob)';

numSites = length(GT);
for s=1:numSites

    % ground truth
    gt = GT(s);

    % process only non-zero classes
    if(gt>0)

        % nn
        c = C(s);
    
        % store it
        NumCgivenL(c,gt) = NumCgivenL(c,gt) + 1; 
    end

end

end