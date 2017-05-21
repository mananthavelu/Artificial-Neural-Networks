load bodyfat_dataset
net = feedforwardnet(20);
[net, tr] = train(net, bodyfatInputs, bodyfatTargets);