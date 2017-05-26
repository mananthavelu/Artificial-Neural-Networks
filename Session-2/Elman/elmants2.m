
% In this script an elman network is trained and tested in order to model a so called Hammerstein model. 
% The system is described like this:

% x(t+1) = 0.6x(t-1) + sin(u(t))
% y(t) = x(t);

% Elman network should be able to understand the relation between output
% y(t) and input u(t). x(t) is a latent variable representing the internal
% state of the system/

% Ricardo Castro-Garcia Feb-2017

%% Clean the workspace
clc;
clear;
close all;

%% Set the parameters of the run
n_tr = 300;             % Number of training points (this includes training and validation).
n_te = 200;             % Number of test points
n_neurons = 50;         % Number of neurons
n = 1000;               % Total number of samples
ne = 500;               % Number of epochs
perc_training = 0.7;    % Number between 0 and 1. The validation set will be 1-perc_training.

if n < n_tr+n_te
    n = n_tr+n_te;
end

if perc_training >= 1 || perc_training <= 0
    error('The training set is ill defined. The variable perc_training should be between 0 and 1')
end

%% Create the samples
% Allocate memory
u = zeros(1, n);
x = zeros(1, n);
y = zeros(1, n);

% Initialize u, x and y
u(1)=randn; 
x(1)=rand+sin(u(1));
y(1)=.6*x(1);

% Calculate the samples
for i=2:n
    u(i)=randn;
    x(i)=.6*x(i-1)+sin(u(i));
    y(i)=x(i);
end

%% Create the datasets
% Training set
X=num2cell(u(1:n_tr)); 
T=num2cell(y(1:n_tr));

% Test set
T_test=num2cell(y(end-n_te:end)); 
X_test=num2cell(u(end-n_te:end));

%% Train and simulate the network
% Create the net and apply the selected parameters
net = newelm(X,T,n_neurons);        % Create network
net.trainParam.epochs = ne;         % Number of epochs
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 1-perc_training;
net.divideParam.trainRatio = perc_training;

net = train(net,X,T);               % Train network

T_test_sim = sim(net,X_test);       % Test the network

%% Plot results
%Plot results and calculate correlation coefficient between target and
%output

figure;
subplot 211
n_test = size(X_test,2);
plot(0:(n_test-1),cell2mat(T_test),'r',0:(n_test-1),cell2mat(T_test_sim),'b');
xlabel('time');
ylabel('y');
legend('target','prediction','Location', 'southeast');
R = corrcoef(cell2mat(T_test),cell2mat(T_test_sim));
R = R(1,2);
my_MSE = mse(cell2mat(T_test)-cell2mat(T_test_sim));
title(['R = ' num2str(R) '. MSE = ' num2str(my_MSE)])
xlim([0, n_test-1])
subplot 212, 
plot(cell2mat(T_test),cell2mat(T_test_sim),'or',cell2mat(T_test),cell2mat(T_test),'.b');
title('Scatter plot')
xlabel('Target')
ylabel('Prediction')
legend('Actual fit', 'Perfect fit', 'location', 'southeast')
%--------------------------------------------------------------------------
