function labels = meanInferenceSolver(unaryClassProb,pairwiseSameProb,adjacency,lambda)
%solve CRF using mean inference with the UGM library
%Dependencies: UGM http://www.cs.ubc.ca/~schmidtm/Software/UGM.html

numClasses = size(unaryClassProb,1);
edgeStruct = UGM_makeEdgeStruct(adjacency,numClasses,true);

nodePot = unaryClassProb';
nEdges = size(edgeStruct.edgeEnds,1);
edgePot = zeros(numClasses,numClasses,nEdges);

for edge = 1:nEdges
    site1 = edgeStruct.edgeEnds(edge,1);
    site2 = edgeStruct.edgeEnds(edge,2);
    sameProb = pairwiseSameProb(site1,site2);
    samePot = sameProb;
    diffPot = 1 - sameProb;
    edgePot(:,:,edge) = lambda*(diffPot * ...
        ones(numClasses) + ...
        (samePot - diffPot) * eye(numClasses));
end

labels = UGM_Decode_MaxOfMarginals(exp(nodePot),exp(edgePot),...
                                edgeStruct,@UGM_Infer_MeanField);


end