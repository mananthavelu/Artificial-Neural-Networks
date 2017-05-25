clear
clc
close all
% With Noise
%


%%%%%%%%%%%
%algorlm.m
% A script comparing performance of different algorithms
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
% traingda
% traincgf
% traincgp
% trainbfg
%%%%%%%%%%%

%generation of examples and targets

xx=0:0.2:4*pi;
xxx=0.05*(rand(1,numel(xx)));% Adding random noise to the data
x=xx+xxx;
y=sin(x);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks

net1=feedforwardnet(20,'trainbr');
net2=feedforwardnet(20,'traingd');
net3=feedforwardnet(20,'traingda');
net4=feedforwardnet(20,'traincgf');
net5=feedforwardnet(20,'traincgp');
net6=feedforwardnet(20,'trainbfg');

%set the same weights and biases for the networks
[net1.iw{1,1},net2.iw{1,1},net3.iw{1,1},net4.iw{1,1},net5.iw{1,1}]=deal(net6.iw{1,1});   
[net1.iw{2,1},net2.iw{2,1},net3.iw{2,1},net4.iw{2,1},net5.iw{2,1}]=deal(net6.iw{2,1});

[net1.b{1},net2.b{1},net3.b{1},net4.b{1},net5.b{1}]=deal(net6.b{1}); 
[net1.b{2},net2.b{2},net3.b{2},net4.b{2},net5.b{2}]=deal(net6.b{2});

%training and simulation
%set the number of epochs for the training
net1.trainParam.epochs=1;  
net2.trainParam.epochs=1;
net3.trainParam.epochs=1;
net4.trainParam.epochs=1;
net5.trainParam.epochs=1;
net6.trainParam.epochs=1;

%train the networks
tic
net1=train(net1,p,t);   
net1_time_1=toc;

tic
net2=train(net2,p,t);
net2_time_1=toc;

tic
net3=train(net3,p,t);
net3_time_1=toc;

tic
net4=train(net4,p,t);
net4_time_1=toc;

tic
net5=train(net5,p,t);
net5_time_1=toc;

tic
net6=train(net6,p,t);
net6_time_1=toc;

training_time_1=[net1_time_1,net2_time_1,net3_time_1,net4_time_1,net5_time_1,net6_time_1];
a11=sim(net1,p); a21=sim(net2,p);a31=sim(net3,p); a41=sim(net4,p);a51=sim(net5,p); a61=sim(net6,p);  % simulate the networks with the input vector p

net1.trainParam.epochs=14;
net2.trainParam.epochs=14;
net3.trainParam.epochs=14;
net4.trainParam.epochs=14;
net5.trainParam.epochs=14;
net6.trainParam.epochs=14;

tic
net1=train(net1,p,t);
net1_time_14=toc;

tic
net2=train(net2,p,t);
net2_time_14=toc;

tic
net3=train(net3,p,t);
net3_time_14=toc;

tic
net4=train(net4,p,t);
net4_time_14=toc;

tic
net5=train(net5,p,t);
net5_time_14=toc;

tic
net6=train(net6,p,t);
net6_time_14=toc;

training_time_14=[net1_time_14,net2_time_14,net3_time_14,net4_time_14,net5_time_14,net6_time_14];
a12=sim(net1,p); a22=sim(net2,p);a32=sim(net3,p); a42=sim(net4,p);a52=sim(net5,p); a62=sim(net6,p);

%For Epochs = 985
net1.trainParam.epochs=985;
net2.trainParam.epochs=985;
net3.trainParam.epochs=985;
net4.trainParam.epochs=985;
net5.trainParam.epochs=985;
net6.trainParam.epochs=985;

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

tic
[net3,tr]=train(net3,p,t);
net3_time_985=toc;
mse_3_985=tr.perf;
len_3_985=length(mse_3_985);

tic
[net4,tr]=train(net4,p,t);
net4_time_985=toc;
mse_4_985=tr.perf;
len_4_985=length(mse_4_985);

tic
[net5,tr]=train(net5,p,t);
net5_time_985=toc;
mse_5_985=tr.perf;
len_5_985=length(mse_5_985);

tic
[net6,tr]=train(net6,p,t);
net6_time_985=toc;
mse_6_985=tr.perf;
len_6_985=length(mse_6_985);

training_time_985=[net1_time_985,net2_time_985,net3_time_985,net4_time_985,net5_time_985,net6_time_985];
plot(1:len_1_985,mse_1_985,1:len_2_985,mse_2_985,1:len_3_985,mse_3_985,1:len_4_985,mse_4_985,1:len_5_985,mse_5_985,1:len_6_985,mse_6_985)
legend('trainlm','traingd','traingda','traincgf','traincgp','trainbfg');
a13=sim(net1,p); a23=sim(net2,p);a33=sim(net3,p); a43=sim(net4,p);a53=sim(net5,p); a63=sim(net6,p);


% Curve fitting plots
f1=figure;
subplot(3,2,1);
plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g',x,cell2mat(a31),'y',x,cell2mat(a41),'m',x,cell2mat(a51),'b',x,cell2mat(a61),'c'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainlm','traingd','traingda','traincgf','traincgp','trainbfg');

subplot(3,2,3); 
plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g',x,cell2mat(a32),'y',x,cell2mat(a42),'m',x,cell2mat(a52),'b',x,cell2mat(a62),'c');
title('15 epochs');
legend('target','trainlm','traingd','traingda','traincgf','traincgp','trainbfg');

subplot(3,2,5);
plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g',x,cell2mat(a33),'y',x,cell2mat(a43),'m',x,cell2mat(a53),'b',x,cell2mat(a63),'c');
title('1000 epochs');
legend('target','trainlm','traingd','traingda','traincgf','traincgp','trainbfg');
xlabel('Input values - p','FontSize',10)

% R Value plots
% For Epochs =1
f2=figure;
subplot(5,5,1);
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result

subplot(5,5,2);
postregm(cell2mat(a21),y);

subplot(5,5,3);
postregm(cell2mat(a31),y);

subplot(5,5,4);
postregm(cell2mat(a41),y);

subplot(5,5,5);
postregm(cell2mat(a51),y);

subplot(5,5,6);
postregm(cell2mat(a61),y);

% For Epochs =14
subplot(5,5,7);
postregm(cell2mat(a12),y);

subplot(5,5,8);
postregm(cell2mat(a22),y);

subplot(5,5,9);
postregm(cell2mat(a32),y);

subplot(5,5,10);
postregm(cell2mat(a42),y);

subplot(5,5,11);
postregm(cell2mat(a52),y);

subplot(5,5,12);
postregm(cell2mat(a62),y);

% For Epochs =985
subplot(5,5,13);
postregm(cell2mat(a13),y);

subplot(5,5,14);
postregm(cell2mat(a23),y);

subplot(5,5,15);
postregm(cell2mat(a33),y);

subplot(5,5,16);
postregm(cell2mat(a43),y);

subplot(5,5,17);
postregm(cell2mat(a53),y);

subplot(5,5,18);
postregm(cell2mat(a63),y);