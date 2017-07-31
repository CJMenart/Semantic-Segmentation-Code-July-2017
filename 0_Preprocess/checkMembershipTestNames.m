function [ memInd ] = checkMembershipTestNames(imageName, testNames)

memInd=0;
for i=1:length(testNames)
    if(strcmp(imageName,testNames{i}))
        memInd=i;
        break;
    end
end

end