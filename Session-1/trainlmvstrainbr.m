clear
clc
close all
% Without noise
%%%%%%%%%%%
% A script comparing performance of different algorithms
% % trainlm - Levenberg - Marquardt
% trainbr -BFGS quasi newton algorithm
%%%%%%%%%%%

%generation of examples and targets

x=0:0.2:4*pi;
y=sin(x);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

xtest=0.01:0.2:4*pi;
ytest=sin(x);
xtest1=con2seq(xtest); ytest1=con2seq(ytest);
%creation of networks

net1=feedforwardnet(20,'trainlm');
net2=feedforwardnet(20,'trainbr');

%setting the same weights for all the networks
net1.iw{1,1}=deal(net2.iw{1,1});   
net1.iw{2,1}=deal(net2.iw{2,1});

%setting the same weights for all the networks
net1.b{1}=deal(net2.b{1}); 
net1.b{2}=deal(net2.b{2});

%training and simulation
%setting the number of epochs for the training as 1
net1.trainParam.epochs=1;  
net2.trainParam.epochs=1;

%train the networks
tic
net1=train(net1,p,t);   
net1_time_1=toc;

tic
net2=train(net2,p,t);
net2_time_1=toc;

training_time_1=[net1_time_1,net2_time_1];
a11=sim(net1,xtest1); a21=sim(net2,xtest1); % simulate the networks with the input vector p

%Setting the epoch to 14 for all the networks
net1.trainParam.epochs=14;
net2.trainParam.epochs=14;

tic
net1=train(net1,p,t);
net1_time_14=toc;

tic
net2=train(net2,p,t);
net2_time_14=toc;

training_time_14=[net1_time_14,net2_time_14];
a12=sim(net1,xtest1); a22=sim(net2,xtest1);

%For Epochs = 985
net1.trainParam.epochs=985;
net2.trainParam.epochs=985;

tic
[net1,tr]=train(net1,p,t);
net1_time_985=toc;
mse_1_985=tr.perf;
len_1_985=length(mse_1_985);

tic
[net2, tr]=train(net2,p,t);
net2_time_985=toc;
mse_2_985=tr.perf;
len_2_985=length(mse_2_985);

training_time_985=[net1_time_985,net2_time_985];
plot(1:len_1_985,mse_1_985,1:len_2_985,mse_2_985)
legend('trainlm','trainbr');
xlabel('Epoch');
ylabel('Performance measure - MSE');
a13=sim(net1,xtest1); a23=sim(net2,xtest1);

% Curve fitting plots
f1=figure;
subplot(3,2,1);
plot(x,y,'bx',xtest,cell2mat(a11),'r',xtest,cell2mat(a21),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainlm','trainbr');

subplot(3,2,3); 
plot(x,y,'bx',xtest,cell2mat(a12),'r',xtest,cell2mat(a22),'g');
title('15 epochs');
legend('target','trainlm','trainbr');

subplot(3,2,5);
plot(x,y,'bx',xtest,cell2mat(a13),'r',xtest,cell2mat(a23),'g');
title('1000 epochs');
legend('target','trainlm','trainbr');
xlabel('Input values - p','FontSize',10)

% R Value plots
% For Epochs =1
epoch_1_r=[];
f2=figure;
subplot(3,3,1);
[m,b,r]=postregm(cell2mat(a11),ytest); % perform a linear regression analysis and plot the result
epoch_1_r=[epoch_1_r,r*r];

subplot(3,3,2);
[m,b,r]=postregm(cell2mat(a21),ytest);
epoch_1_r=[epoch_1_r,r*r];

% For Epochs =14
epoch_14_r=[];
subplot(3,3,3);
[m,b,r]=postregm(cell2mat(a12),ytest);
epoch_14_r=[epoch_14_r,r*r];

subplot(3,3,4);
[m,b,r]=postregm(cell2mat(a22),ytest);
epoch_14_r=[epoch_14_r,r*r];


% For Epochs =985
epoch_985_r=[];
subplot(3,3,5);
[m,b,r]=postregm(cell2mat(a13),ytest);
epoch_985_r=[epoch_985_r,r*r];

subplot(3,3,6);
[m,b,r]=postregm(cell2mat(a23),ytest);
epoch_985_r=[epoch_985_r,r*r];


%Plotting the R^2 values
epoch_r2=[epoch_1_r;epoch_14_r ; epoch_985_r]
figure;
plot(epoch_r2)
legend('R^2 values')
xlabel('Epoch')
ylabel('R^2 value')

%Plotting the training time for various algorithms
training_time=[training_time_1;training_time_14;training_time_985];
figure;
plot(training_time)
legend('trainlm','trainbr');
xlabel('Epochs')
ylabel('Training time, second')