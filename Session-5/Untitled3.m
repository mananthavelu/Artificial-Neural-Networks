% load('Data_Problem1_regression.mat');

 %r0652832
 %d1=8; d2=6; d3=5; d4=3; d5=2;
 %Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);
 
    data=load('Data_Problem1_regression.mat');
    X1 = data.X1;
    X2 = data.X2;
    sno = sort([0,6,5,2,8,3,2],'descend');
    digits = sno(1:5);
    T=(sum(bsxfun(@times,[data.T1 data.T2 data.T3 data.T4 data.T5],digits),2)/sum(digits));
    N = max(size(T));
    
    trainIdx = randsample(N,1000);
    valIdx = randsample(N,1000);
    testIdx = randsample(N,1000);
    
    Xdata = [X1(trainIdx) X2(trainIdx);X1(valIdx) X2(valIdx);X1(testIdx) X2(testIdx)]';
    Tdata = [T(trainIdx);T(valIdx);T(testIdx)]';

    net=feedforwardnet(10, 'trainlm');
    net.trainParam.epochs=1000;
    net=train(net, [X1(trainIdx) X2(trainIdx)]', T(trainIdx)');
    op = sim(net,[X1(valIdx) X2(valIdx)]');
    error = mse(T(valIdx), op');
    
    
    figure
    title_str = sprintf('%s, %d Hidden units, %d Epochs\nmse = %f', 'trainlm', '10', error);
    surf(([X1(valIdx) X2(valIdx)]), op',title_str);
    %title_str = sprintf('%s, %d Hidden units, %d Epochs', algo{i}, hiddenUnits(k), EpochsArr(ep));
    %title_str = strcat(strip(title_str),'.png');
    %saveas(gcf, strcat('C:\Artificial Neural Networks\Exercises\Session-5', title_str));
    %set(gcf, 'visible', 'off');
    %cumEpoch = cumEpoch + EpochsArr(ep);
    