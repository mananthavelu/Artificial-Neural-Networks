function regress

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
    
    %load('Data_Problem1_regression.mat');

    %r0652832
    %d1=8; d2=6; d3=5; d4=3; d5=2;
    %Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);
    
    %splitPoint = 0.8*length(Tnew);
    
    trainset = T(trainIdx);
    valset = T(valIdx);

    Xtrain = [X1(trainIdx) X2(trainIdx)];
    Xval = [X1(valIdx) X2(valIdx)]; size(Xtrain)
    
    figure, plotSurfaceQ1(X1(trainIdx), X2(trainIdx), trainset,'Training Set');
    figure, plotSurfaceQ1(X1(valIdx), X2(valIdx), valset,'Test Set');

    algo = {'traingd', 'trainlm', 'trainbr'};
    %algo = {'trainlm'};
    hiddenUnits = [10, 100];
    EpochsArr = [1000];

    for i=1:length(algo)
        for k=1:length(hiddenUnits)
            cumEpoch = 0;
            for ep=1:length(EpochsArr)
                net=feedforwardnet(hiddenUnits(k), algo{i});
                net.trainParam.epochs=EpochsArr(ep) - cumEpoch;
                net=train(net, Xtrain', trainset');
                op = sim(net,Xval');
                error = mse(valset, op');
                figure
                title_str = sprintf('%s, %d Hidden units, %d Epochs\nmse = %f', algo{i}, hiddenUnits(k), EpochsArr(ep), error);
                plotSurfaceQ1(X1(valIdx), X2(valIdx), op',title_str);
                title_str = sprintf('%s, %d Hidden units, %d Epochs', algo{i}, hiddenUnits(k), EpochsArr(ep));
                title_str = strcat(strip(title_str),'.png');
                saveas(gcf, strcat('C:\Artificial Neural Networks\Exercises\Session-5', title_str));
                %set(gcf, 'visible', 'off');
                cumEpoch = cumEpoch + EpochsArr(ep);
            end
        end
    end

end

function h=plotSurfaceQ1(X1,X2,T,titleStr)
    XVEC = X1;
    YVEC = X2;
    ZVEC = T;
    size(X1)
    size(X2)
    size(T)
    F=scatteredInterpolant(XVEC,YVEC,ZVEC);
    [Xq,Yq]=meshgrid(min(XVEC):0.01:max(XVEC),min(YVEC):0.01:max(YVEC));
    Vq = F(Xq,Yq);
    h=surfc(Xq,Yq,Vq);
    xlabel('X_1');
    ylabel('X_2');
    zlabel('T');
    title(titleStr);
end

function title_str = strip(title_str)
    title_str = strrep(title_str, '=', '');
    title_str = strrep(title_str, '\n', '');
    title_str = strrep(title_str, ' ', '');
    title_str = strrep(title_str, ',', '');
    title_str = strrep(title_str, '(', '');
    title_str = strrep(title_str, ')', '');
end