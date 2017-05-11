function [V, newX, D] = pca_function(X)
    X = bsxfun(@minus, X, mean(X,1));           %# Zero-centering with mean
    C = (X'*X)./(size(X,1)-1);                  %'# Computing the covariance matriz of given data (X)

    [V,D] = eig(C);
    [D, order] = sort(diag(D), 'descend');       %# sort cols high to low
    V = V(:,order);

    newX = X*V(:,1:end);
end