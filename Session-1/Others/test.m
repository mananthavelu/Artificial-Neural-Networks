p = [-1:.05:1];
t = sin(2*pi*p)+0.1*randn(size(p));
net = feedforwardnet(2,'trainbr');
net = train(net,p,t);
a = net(p);