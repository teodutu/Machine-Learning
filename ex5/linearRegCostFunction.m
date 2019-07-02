function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  % LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  % regression with multiple variables
  % [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  % cost of using theta as the parameter for linear regression to fit the 
  % data points in X and y. Returns the cost in J and the gradient in grad

  [m n] = size(X); % number of training examples and that of features

  % Calculate the predictions
  h = X * theta;

  % First calculate the unregularised cost
  J_unreg = sum((h - y).^2) / (2 * m);
  % Then regularise it by the parameter lambda
  J = J_unreg + lambda * sum(theta(2:n).^2) / (2*m);;

  % Similarly, first, compute the unregularised gradient of the cost function
  grad_unreg = X' * (h - y) / m;
  grad = grad_unreg;
  % Apply regularisation to the gradient, except for the first partial derivative
  grad(2:n) += lambda * theta(2:n) / m;
end
