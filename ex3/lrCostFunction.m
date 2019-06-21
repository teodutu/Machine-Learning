function [J, grad] = lrCostFunction(theta, X, y, lambda)
  % LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  % regularization
  % J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  % theta as the parameter for regularized logistic regression and the
  % gradient of the cost w.r.t. to the parameters. 

  % Copy-pasted from the previous exercise
  [m n] = size(X); % number of training examples and that of features

  h = sigmoid(X * theta);

  J_unreg = -1 * (y' * log(h) + (1 - y)' * log(1 - h)) / m;
  J = J_unreg + lambda * sum(theta(2:n).^2) / (2*m);;

  grad_unreg = X' * (h - y) / m;
  grad = grad_unreg;
  grad(2:n) += lambda * theta(2:n) / m;
end
