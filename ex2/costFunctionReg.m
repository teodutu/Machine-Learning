function [J, grad] = costFunctionReg(theta, X, y, lambda)
  % COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  % J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  % theta as the parameter for regularized logistic regression and the
  % gradient of the cost w.r.t. to the parameters. 

  % Initialize some useful values
  [m n] = size(X); % number of training examples and number of features

  % Compute the unregularised J and gradient
  [J grad] = costFunction(theta, X, y);

  % Regularise J and gradient by using the parameter lambda
  J += lambda * sum(theta(2:n).^2) / (2*m);
  grad(2:n) += lambda * theta(2:n) / m;
end
