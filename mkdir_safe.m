function [] = mkdir_safe(dir)
%wraps makedir with a safey check, and makes if does not exist
if(~exist(dir,'dir'))
    if(~mkdir(dir))
    	fprintf(1,'Error creating %s... quitting\n', dir);
        pause;
        return;  
    end
end

end
