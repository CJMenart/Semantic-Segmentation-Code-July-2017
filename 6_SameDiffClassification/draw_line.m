function im = draw_line(im,centroid1,centroid2,color)

    x1 = centroid1(1);
    y1 = centroid1(2);
    x2 = centroid2(1);
    y2 = centroid2(2);
    
    sz = size(im);
    lineMat = zeros(sz(1),sz(2),'logical');
    
    x1 = double(x1);
    x2 = double(x2);
    y1 = double(y1);
    y2 = double(y2);
    
    %in order to make sure a 'filled in' line gets drawn, we draw our line
    %across the dimension with the largest change. This also handles the
    %vertical-line case
    if (abs(x2 - x1) > abs(y2 - y1)) %case of a more horizontal line
    
        a = (y2-y1)/(x2-x1);
        b = y1 - x1*a;

        x = linspace(x1,x2,abs(x1-x2)+1);
        y = round(x*a + b);
    elseif x2==x1 && y2==y1
        x = x2;
        y = y1;
    else %case of a more vertical line
        
        a = (x2-x1)/(y2-y1);
        b = x1 - y1*a;
    
        y = linspace(y1,y2,abs(y1-y2)+1);
        x = round(y*a + b);
    end
    
    lineMat(sub2ind(size(lineMat), y, x)) = 1;
    
    for chan = 1:sz(3)
        channel = im(:,:,chan);
        channel(lineMat) = color(chan);
        im(:,:,chan) = channel;
    end
end