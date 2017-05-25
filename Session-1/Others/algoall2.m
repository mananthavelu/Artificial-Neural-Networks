function algoall2(xtrain,ytrain,h,npower,index,dataset)
    ypower = sqrt(sum(ytrain.^2));
    nytrain = ytrain + 0.01*npower*ypower*randn(size(ytrain));
    suffix=sprintf('e2-%d-%d-%d.png',index,h,npower);
    prefix=sprintf('%s h:%d np:%d %s',dataset,h,npower);
    p=con2seq(xtrain); t=con2seq(nytrain); % convert the data to a useful format
    algs = {'trainlm','traingd'};
    epochs = [10,90,900];
    epochstr = {'10','100','1000'};
    net=feedforwardnet(h,'trainbr');
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    result(1).net = net;
    for i=1:max(size(algs))
        nnet=feedforwardnet(h,algs{i});
        nnet.layers{1}.transferFcn = 'tansig';
        nnet.layers{2}.transferFcn = 'purelin';
        nnet.trainParam.showWindow = false;
        nnet.trainParam.showCommandLine = false;
        nnet.iw{1,1}=net.iw{1,1};
        nnet.lw{2,1}=net.lw{2,1};
        nnet.b{1}=net.b{1};
        nnet.b{2}=net.b{2};
        result(i+1).net = nnet;
    end
    c = {'-or','-+g','-sb'};
    for j=1:max(size(epochs))
        figure, hold on;
        plot(xtrain,ytrain,'-x'); % plot the sine function and the output of the networks
        if npower > 0
            plot(xtrain,ytrain,'-+'); % plot the sine function and the output of the networks
        end
        for i=1:max(size(result))
            result(i).net.trainParam.epochs=epochs(j);  % set the number of epochs for the training 
            result(i).net=train(result(i).net,p,t);   % train the networks
            a=cell2mat(sim(result(i).net,p)); 
            plot(xtrain,a,c{i}); % plot the sine function and the output of the networks    
        end
        xlabel('x');
        ylabel(dataset);
        title(sprintf('%s %s epochs',prefix,epochstr{j}));
        if npower > 0
            legend('target','noisy','traingd','trainlm','trainbfg');
        else
            legend('target','trainbr','trainlm','traingd');
        end
        hold off;
        set(gcf,'visible','off') 
        saveas(gcf,sprintf('%s-%s',epochstr{j},suffix));
    end
end