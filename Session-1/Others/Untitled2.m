[x, t] = bodyfat_dataset;
net = feedforwardnet(20);
net = train(net, x, t);
y = net(x);