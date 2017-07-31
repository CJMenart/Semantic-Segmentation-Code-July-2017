function [err] = mycostfun(P_L_lim, In, Out, ProbCgivenL, validTruthLabels)

numExamples = size(In,2);
numLabels = size(In,1);

% make priors from limited (non-zero) solved priors
P_L = zeros(numLabels,1);
P_L(validTruthLabels) = P_L_lim;

% ground truth result
h = vec2ind(Out);

% make mapping matrix
MM = zeros(numLabels,numLabels);
for i=1:numLabels
    for j=1:numLabels
        MM(i,j) = ProbCgivenL(j,i)*P_L(i);
    end
end
for j=1:numLabels
    v=MM(:,j);
    sv = sum(v);
    if(sv>0)
        v=v/sv;
    end
    MM(:,j)=v;
end

% do the mapping
tmpOut = MM*In;

% get error function (cross-entropy)
err = 0;
for j=1:numExamples
    err = err - log(max(tmpOut(h(j),j),1e-8));
end


end

