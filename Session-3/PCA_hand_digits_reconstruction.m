load threes -ASCII

%Applying PCA
[V, newX, D] = pca_function(threes);

%Plotting the eigen values
figure
plot(D)
xlabel('Data points');
ylabel('Eigen values');

%Reconstruction of the data
mu = mean(threes);
figure;
colormap('gray')
imagesc(reshape(mu(1,:),16,16),[0,1]);
xhat = bsxfun(@minus,threes,mu); % subtract the mean

%Constructed Original data
norm(newX * V' - xhat);

%Reconstruction of original image for first 4 components
figure
title('Reconstructed images')
fig=1;
for i=1:4  
    Xapprox = newX(:,1:i) * V(:,1:i)';
    Xapprox = bsxfun(@plus,mu,Xapprox); % add the mean back in
        
    %Reconstructed image
    subplot(2,2,fig)
    colormap('gray')
    imagesc(reshape(Xapprox(1,:),16,16),[0,1])
    title(i);
    xlim auto
    ylim auto
    fig=fig+1;
end

%Reconstruction error
Error=[];
for i=1:50    
    Xapprox = newX(:,1:i) * V(:,1:i)';
    Xapprox = bsxfun(@plus,mu,Xapprox); % add the mean back in
    err=threes-Xapprox;
    val=norm(err,2)^2;
    Error = [Error val];
end

figure
plot(1:50,Error);
title('Reconstruction Error');
xlabel('Eigen Value')
ylabel('Error')

cumEigVal = cumsum(D/sum(D));
cumEigVal_50=cumEigVal(1:50);


figure;
title('Reconstruction error vs  Principal components vs Cumsum')
yyaxis left
plot(1:50,Error)
xlabel('Principal components')
ylabel('Reconstruction error')

yyaxis right
plot(1:50,cumEigVal_50')
ylabel('Cumsum')

