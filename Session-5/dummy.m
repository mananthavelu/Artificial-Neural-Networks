clear;
clc;
load('Data_Problem1_regression.mat');
Tnew = (8*T1 + 6*T2 + 6*T3 +5*T4 + 4*T5)/(8+6+6+5+4);
data = [X1 X2 Tnew];

train = datasample(data,1000);
val = datasample(data,1000);
test = datasample(data,1000);

% figure, plotSurface(train,'Training Set Surface');
% figure, plotSurface(val,'Validation Set Surface');
% figure, plotSurface(test,'Test Set Surface');

Xdata = [train(:,1) train(:,2);val(:,1) val(:,2);test(:,1) test(:,2)]';
Tdata = [train(:,3);val(:,3);test(:,3)]';

hiddenUnits = [10 50 100];
perf = zeros(max(size(hiddenUnits)),1);
for i=1:max(size(hiddenUnits))
    [out(i).net, out(i).tr] = designNN(Xdata,Tdata,hiddenUnits(i));
    perf(i) = out(i).tr.best_vperf;
end
[~,hid] = min(perf);
finalNN = out(hid);

Tpred = sim(finalNN.net,[test(:,1) test(:,2)]')';

% figure,plotSurface([test(:,1) test(:,2) Tpred],'Surface of the Approximation');
% figure,plotperform(finalNN.tr);
% figure,plotregression(test(:,3)', Tpred', 'Test Set Correlations');

function [net,tr]= designNN(X,T,totalHidden)
    net = feedforwardnet(totalHidden);
    net.divideFcn ='divideind';
    net.divideParam.trainInd = 1:1000;
    net.divideParam.valInd = 1001:2000;
    net.divideParam.testInd = 2001:3000;
    %net.trainParam.showWindow=0;
    %net.trainParam.epochs=10000;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn ='purelin';
    [net,tr]=train(net,X,T);
end

function [fig] = plotSurface(data,type)
    X1 = data(:,1); X2 = data(:,2); T = data(:,3);
    F=scatteredInterpolant(X1,X2,T);

    [X,Y]=meshgrid(min(X1):0.01:max(X1),min(X2):0.01:max(X2));
    Z = F(X,Y);
    fig = surfc(X,Y,Z);
    xlabel('X_1');
    ylabel('X_2');
    zlabel('T');
    title(type);
end