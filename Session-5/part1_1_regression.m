clear;
clc;
close all

%Load the dataset
load('Data_Problem1_regression.mat');

%Create the target variables
%Student Number : 0652832
Tnew = (8*T1 + 6*T2 + 5*T3 +3*T4 + 2*T5)/(8+6+5+3+2);%r0652832 -> 8 6 5 3 2

%Creating the dataset
data = [X1 X2 Tnew];

%Split the dataset into Training, Validation and Testing data by randomly
%sampling with replacement
train = datasample(data,1000);
val = datasample(data,1000);
test = datasample(data,1000);

%Surface plots for the training, validation and testing data
figure, plotSurface(train,'Training Set Surface');
figure, plotSurface(val,'Validation Set Surface');
figure, plotSurface(test,'Test Set Surface');

%Splitting into training, validation and test data
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

figure,plotSurface([test(:,1) test(:,2) Tpred],'Surface of the Approximation');
figure,plotperform(finalNN.tr);
figure,plotregression(test(:,3)', Tpred', 'Test Set Correlations');

%Designing the required network
function [net,tr]= designNN(X,T,totalHidden)
    net = feedforwardnet(totalHidden);
    net.divideFcn ='divideind';
    net.divideParam.trainInd = 1:1000;
    net.divideParam.valInd = 1001:2000;
    net.divideParam.testInd = 2001:3000;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn ='purelin';
    [net,tr]=train(net,X,T);
end
%Plots
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