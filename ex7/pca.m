function [U, S] = pca(X)
  % PCA Run principal component analysis on the dataset X
  % [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
  % Returns the eigenvectors U, the eigenvalues (on diagonal) in S

  % Useful values
  m = size(X, 1);

  % Calculate the covariance matrix
  Sigma = X' * X / m;

  % Get the singular values and the left and right singular vectors
  [U S V] = svd(Sigma);
end