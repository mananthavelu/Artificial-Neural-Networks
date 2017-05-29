clear;
clc;
warning off;
setdemorandstream(8);
dataset = importdata('winequality-white.csv');

%Creating the classes for the given dataset
reqIndex = (dataset.data(:,12)==5 | dataset.data(:,12)==7);
Xdata=dataset.data(reqIndex,1:(end-1))';
Y=dataset.data(reqIndex,end);
Y(Y~=7) = 0;
Y(Y==7) = 1;
Ydata = Y';
%Ydata = [Y ~Y]';

[trainInd,~,testInd] = dividerand(size(Xdata,2),0.8,0,0.20);

Xtrain = Xdata(:,trainInd);
Ytrain = Ydata(:,trainInd);
Xtest = Xdata(:,testInd);
Ytest = Ydata(:,testInd);

hiddenUnits = [10 50 100];
perf = zeros(max(size(hiddenUnits)),1);

for i=1:max(size(hiddenUnits))
    [out(i).net, out(i).tr] = designNN(Xtrain,Ytrain,hiddenUnits(i));
    perf(i) = out(i).tr.best_vperf;
end

[~,hid] = min(perf);
finalNN = out(hid);
ccr = computeCCR(finalNN.net,Xtest,Ytest)
figure,plotperform(finalNN.tr);

% %% part 2:
%Performing dimensionality reduction technique
eigVect = PCA(Xtrain);

hiddenUnits = [10 50 100];
%perf = zeros( size(Xtrain,1) * max(size(hiddenUnits)),1);

for j=2:size(Xtrain,1)
    for i=1:max(size(hiddenUnits))
        
       [rXtrain,rXtest] = reconstruct(Xtrain,Xtest,eigVect,j);
        
        [out(i,j-1).net, out(i,j-1).tr] = designNN(rXtrain,Ytrain,hiddenUnits(i));
        perf(i,j-1) = out(i,j-1).tr.best_vperf;
        ccr_PCA(i,j-1) = computeCCR(out(i,j-1).net,rXtest,Ytest);
    end
end
%[~,hid] = min(perf);
[~, hid] = min(perf(:));
[Nhid,Npca] = ind2sub(size(perf),hid);
Npca+1
finalNN = out(Nhid,Npca);
ccr_PCA(Nhid,Npca)
figure,plotperform(finalNN.tr);


%% function %%

function [rXtrain,rXtest] = reconstruct(Xtrain,Xtest,W,nc)
    X = Xtrain';
    ZX = bsxfun(@minus, X, mean(X));
    Xt = Xtest';
    Ztest = bsxfun(@minus, Xt, mean(Xt));
    W = W(:,1:nc);
    rXtrain = (ZX*W)';
    rXtest = (Ztest*W)';
end

function [net,tr]= designNN(X,Y,totalHidden)
    net = patternnet(totalHidden);
    net.divideFcn ='dividerand';
    net.divideParam.trainInd = 85/100;
    net.divideParam.valInd = 15/100;
    net.divideParam.testInd = 0/100;
    net.trainParam.showWindow=0;
    net.trainParam.epochs=10000;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'softmax';
    net.performFcn ='crossentropy';
    [net,tr]=train(net,X,Y);
    view(net)
end

function [ccr] = computeCCR(net,Xtest,Ytest)
    Ypred = net(Xtest);
    cmat=confusionmat(Ypred(1,:)>0.5,Ytest(1,:)>0.5);
    ccr = (sum(diag(cmat))/sum(sum(cmat))) * 100;
end

function eigVect = PCA(Xtrain)
    X = Xtrain';
    X = bsxfun(@minus, X, mean(X));
    
    C = cov(X);
    [eigVect,eigVal] = eig(C);
    
    eigVal = diag(eigVal);
    [eigVal,IDX]=sort(eigVal,'descend');
    eigVect = eigVect(:,IDX);
    
    figure, hold on;
    %cum = cumsum(eigVal);
    plot(eigVal);
    xlabel('Index');
    ylabel('Eigenvalue');
    title('PCA Eigenvalue Plot');
    hold off;
    
    cumEigVal = cumsum(eigVal/sum(eigVal));
    figure, hold on;
    plot(cumEigVal);
    xlabel('Number of Principal Components');
    ylabel('Fraction of Variance Captured');
    title('PCA Variance Plot');
    hold off;
    
end