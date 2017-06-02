clear;
clc;
setdemorandstream(8);

patterns = getCharacterPatterns();
plotPatterns(patterns,'Character Dataset');

%Retrieval of 5 patterns

 T = patterns(1:5,:);
 distortedT = getDistortedPattern(T);
 
 %Creation of Hopfield neural network
 net=newhop(T');
 
 % Original image reconstruction
 [Y,~,~] = sim(net,{size(distortedT,1) 3},[],distortedT');
 Y=Y{end}';
 Y=round(Y,0);
 
 error=computeError(Y,T);
 
 plotPatterns(T,'Original Characters');
 plotPatterns(distortedT,'Distorted Characters');
 plotPatterns(Y,sprintf('Recollected Characters'));


%Error vs Number of patterns
 noOfIter = 1;
 errors = zeros(size(patterns,1),1);
 
 for j=1:size(patterns,1)
 
     T = patterns(1:j,:);
     distortedT = getDistortedPattern(T);
     net=newhop(T');
     [Y,~,~] = sim(net,{size(distortedT,1) noOfIter},[],distortedT');
     Y=Y{end}';
     Y=sign(Y);
     errors(j)=computeError(Y,T);
 end

 
 N = size(patterns,2);
 cap = N/(4*log(N));
 figure, plot(errors);
 hold on
 plot(round(cap),0.5,'r*');
 xlabel('Number of Patterns');
 ylabel('Number of Pixel Errors');
 title('Hopfield Network Storage Capacity');
 hold off

%Retrieval of 25 patterns
noOfIter = 30;
T = patterns(1:25,:);
Tr = resize(T,7,5,5);
distortedT = getDistortedPattern(T);
distortedTr = resize(distortedT,7,5,5);
net=newhop(Tr');

[Yr,~,~] = sim(net,{size(distortedTr,1) noOfIter},[],distortedTr');
Yr=Yr{end}';
Yr=round(Yr,0);

Y = resize(Yr,35,25,1/5);

error=computeError(Y,T);
disp(error)

plotPatterns(T,'Original Characters');
plotPatterns(distortedT,'Distorted Characters');
plotPatterns(Y,sprintf('Recollected Characters'));

%% functions:

function resizedPattern = resize(patterns,ySize,xSize,scaleFactor)
    resizedPattern = zeros(size(patterns,1),round(scaleFactor*scaleFactor*size(patterns,2)));
    
    for i=1:size(patterns,1)
        img = reshape(patterns(i,:),ySize,xSize);
        img(img==-1) = 0;
        resizedimg = imresize(img,scaleFactor);
        resizedimg = round(resizedimg,0);
        reshaped = reshape(resizedimg,1,size(resizedPattern,2));
        reshaped(reshaped==0) = -1;
        resizedPattern(i,:) = reshaped;
    end
end

function error=computeError(patterns,distpatterns)
    errorMat = patterns.*distpatterns;
    errorMat(errorMat==0) = -1;
    errorMat(errorMat>0) = 0;
    errorMat(errorMat<0) = 1;
    error = sum(sum(double(errorMat)));
end

function distPattern = getDistortedPattern(pattern)
    distPattern = pattern;
    for i=1:size(pattern,1)
        distindxs = randsample(size(pattern,2),3);
        distPattern(i,distindxs) = -1*pattern(i,distindxs);
    end
end

function plotPatterns(patterns,titleStr)
    allchars = zeros(7,185);
    
    for i=1:size(patterns,1)
        index = (i-1)*5+1;
        img = reshape(patterns(i,:),7,5);
        img(img==-1) = 0;
        allchars(:, index:(index+4))=img;
    end
    figure,subplot(2,1,1), subimage(allchars(:,1:90));title(titleStr);
    subplot(2,1,2), subimage(allchars(:,91:185));title(titleStr);   
end

function patterns=getCharacterPatterns()
    patterns = zeros(37,35);
    
     m = [0    0    0    0    0;
         1    1    1    1    1;
         1    0    1    0    1;
         1    0    1    0    1;
         1    0    1    0    1;
         1    0    1    0    1;
         1    0    1    0    1];
 
 
    
 
    a = [0    0    0    0    0;
         0    0    0    0    0;
         0    1    1    1    0;
         1    0    0    1    0;
         1    0    0    1    0;
         1    0    0    1    0;
         1    1    1    1    1];
 
     
     
    r = [0    0    0    0    0;
         0    0    0    0    0;
         0    1    1    1    0;
         0    1    0    0    0;
         0    1    0    0    0;
         0    1    0    0    0;
         0    1    0    0    0];
 
     
     
     i = [0    0    0    0    0;
         0    0    1    0    0;
         0    0    0    0    0;
         0    0    1    0    0;
         0    0    1    0    0;
         0    0    1    0    0;
         0    0    1    0    0];
     
     
     u = [0    0    0    0    0;
         0    0    0    0    0;
         0    1    0    1    0;
         0    1    0    1    0;
         0    1    0    1    0;
         0    1    0    1    0;
         0    1    1    1    0];
     
     
     t = [0    1    0    0    0;
         1    1    1    0    0;
         0    1    0    0    0;
         0    1    0    0    0;
         0    1    0    0    0;
         0    1    0    0    0;
         0    1    1    0    0];
     
     h = [1    0    0    0    0;
         1    0    0    0    0;
         1    1    1    1    0;
         1    0    0    1    0;
         1    0    0    1    0;
         1    0    0    1    0;
         1    0    0    1    0];
     n = [0    0    0    0    0;
         0    0    0    0    0;
         1    1    1    1    0;
         1    0    0    1    0;
         1    0    0    1    0;
         1    0    0    1    0;
         1    0    0    1    0];
     
    
    v = [0    0    0    0    0;
         0    0    0    0    0;
         1    0    0    0    1;
         1    0    0    0    1;
         1    1    0    1    1;
         0    1    0    1    0;
         0    0    1    0    0];
 
    e = [0    0    0    0    0;
         0    0    0    0    0;
         1    1    1    1    0;
         1    0    0    1    0;
         1    1    1    1    0;
         1    0    0    0    0;
         1    1    1    1    0];
 
  
    l = [1    0    0    0    0;
         1    0    0    0    0;
         1    0    0    0    0;
         1    0    0    0    0;
         1    0    0    0    0;
         1    0    0    0    0;
         1    1    1    1    0];
 
     
    %Lowercase Letters
    lowerchars = [m a r i u t h n v e l];
    
    %Uppercase Letters
    chars = prprob;
    upperchars = zeros(7,size(chars,2)*5);
    
    for i=1:size(chars,2)
        index = (i-1)*5+1;
        upperchars(:,index:(index+4)) = reshape(chars(:,i),5,7)';
    end
        allchars = [lowerchars upperchars];
        
    %Making Pattern Vectors
    for i=0:5:180
        index = round(i/5,0)+1;
        image = double(allchars(:,((i+1):(i+5))));
        image(image == 0) = -1;
        patterns(index,:)=reshape(image,1,35);
    end
end