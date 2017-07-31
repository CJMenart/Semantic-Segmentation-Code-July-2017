function [ MappingMatrix ] = makeMappingMatrix(Prob_CgivenL, Prob_L)

% how many labels
numLabels = size(Prob_CgivenL,1);

% create un-normalized mapping matrix
MappingMatrix = zeros(numLabels, numLabels);
% set probabilities
for i=1:numLabels
    for j=1:numLabels
        MappingMatrix(i,j) = Prob_CgivenL(j,i)*Prob_L(i);
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


end