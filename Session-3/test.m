load threes -ASCII

%Applying PCA
[V, newX, D] = pca_function(threes);
mu = mean(threes);

Xapprox = newX(:,1:4) * V(:,1:4)';
Xapprox = bsxfun(@plus,mu,Xapprox); % add the mean back in


figure
plot(Xapprox(:,1),threes(:,1),'.'); hold on; plot([-0.5 0.5],[-0.5 2])
xlabel('Approximation'); ylabel('Actual value'); grid on;
xlim([-0.2 0.2])
ylim([-0.2 0.2])

Xapprox = newX(:,1) * V(:,1)';
Xapprox = bsxfun(@plus,mu,Xapprox);
plot(Xapprox(:,1),threes(:,1),'.'); hold on; plot([-4 4],[-4 4])
xlabel('Approximation'); ylabel('Actual value'); grid on;

hey=100*D/sum(D);
 
corr(newX(:,1),threes(:,1))