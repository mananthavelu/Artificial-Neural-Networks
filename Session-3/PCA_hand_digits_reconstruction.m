load threes -ASCII

[V, newX, D] = pca_function(threes);

%Plotting the eigen values
figure
plot(D)
xlabel('Data points');
ylabel('Eigen values');

%Reconstruction of the data
mu = mean(threes);
xhat = bsxfun(@minus,threes,mu); % subtract the mean
norm(newX * V' - xhat);

%figure
fig=1;
for i=1:4
    
    %Reconstruction of original image
    Xapprox = newX(:,1:i) * V(:,1:i)';
    Xapprox = bsxfun(@plus,mu,Xapprox); % add the mean back in
    err=sum(((threes-Xapprox).^2)/size(threes,1));
    
    %Reconstructed image
    subplot(2,2,fig)
    imagesc(reshape(Xapprox(1,:),16,16),[0,1])
    %xlim auto
    %ylim auto
        
    %plot(Xapprox(:,1),threes(:,1),'.'); hold on; plot([-4 4],[-4 4])
    %xlabel('Approximation'); ylabel('Actual value'); grid on;
    %xlim auto
    %ylim auto
    fig=fig+1;
end

%figure
%plot(Xapprox(:,1),threes(:,1),'.'); hold on; plot([-4 4],[-4 4])
%xlabel('Approximation'); ylabel('Actual value'); grid on;
%xlim([-0.2 0.2])
%ylim([-0.2 0.2])

%Xapprox = newX(:,1) * V(:,1)';
%Xapprox = bsxfun(@plus,mu,Xapprox);
%plot(Xapprox(:,1),threes(:,1),'.'); hold on; plot([-4 4],[-4 4])
%xlabel('Approximation'); ylabel('Actual value'); grid on;

%hey=100*D/sum(D);
 
%corr(newX(:,1),data(:,1))