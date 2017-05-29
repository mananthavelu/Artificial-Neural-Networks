patterns = getCharacterPatterns();
plotPatterns(patterns,'Characters');

distpatterns = patterns;
for i=1:size(patterns,1)
    distindxs = randsample(size(patterns,2),numbits);
    distpatterns(i,distindxs) = -1*patterns(i,distindxs);
end
plotPatterns(distpatterns,'Distorted Characters (3 bits)')

function plotPatterns(patterns,titleStr)
    allchars = zeros(7,180);
    
    for i=1:size(patterns,1)
        index = (i-1)*5+1;
        img = reshape(patterns(i,:),7,5);
        img(img==-1) = 0;
        allchars(:, index:(index+4))=img;
    end
    
    figure,subplot(3,1,1), subimage(allchars(:,1:60));
    title(titleStr);
    subplot(3,1,2), subimage(allchars(:,61:120));title(titleStr);
    subplot(3,1,3), subimage(allchars(:,121:180));   
end

function patterns=getCharacterPatterns()
    patterns = zeros(31,35);
    
    r =[0    0    0    0    0;
        0    0    0    0    0;
        0    1    1    1    0;
        0    1    0    0    0;
        0    1    0    0    0;
        0    1    0    0    0;
        0    1    0    0    0];
    
    a =[0    0    0    0    0;
        0    0    0    0    0;
        0    1    1    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    1    1    1    1];
    
    m =[0    0    0    0    0;
        0    0    0    0    0;
        0    1    0    1    0;
        1    0    1    0    1;
        1    0    1    0    1;
        1    0    1    0    1;
        1    0    1    0    1];
    
    v =[0    0    0    0    0;
        0    0    0    0    0;
        1    0    0    0    1;
        1    0    0    0    1;
        1    1    0    1    1;
        0    1    0    1    0;
        0    0    1    0    0];
    
    i =[0    0    1    0    0;
        0    0    0    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0];
    
    n =[0    0    0    0    0;
        0    0    0    0    0;
        1    1    1    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    0    0    1    0];
    
    d =[0    0    0    1    0;
        0    0    0    1    0;
        0    0    0    1    0;
        1    1    1    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    1    1    1    1]; 
    
    t =[0    1    0    0    0;
        1    1    1    0    0;
        0    1    0    0    0;
        0    1    0    0    0;
        0    1    0    0    0;
        0    1    0    0    0;
        0    1    1    0    0];
    
    h =[1    0    0    0    0;
        1    0    0    0    0;
        1    1    1    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    0    0    1    0;
        1    0    0    1    0];
    
    l =[0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    0    0;
        0    0    1    1    0];
    
    %Lowercase Letters
    lowerchars = [r a m v i n d t h l];
    
    %Uppercase Letters
    chars = prprob;
    upperchars = zeros(7,size(chars,2)*5);
    
    for i=1:size(chars,2)
        index = (i-1)*5+1;
        upperchars(:,index:(index+4)) = reshape(chars(:,i),5,7)';
    end
        allchars = [lowerchars upperchars];
        
    %Make Pattern Vectors
    for i=0:5:175
        index = round(i/5,0)+1;
        image = double(allchars(:,((i+1):(i+5))));
        image(image == 0) = -1;
        patterns(index,:)=reshape(image,1,35);
    end
end