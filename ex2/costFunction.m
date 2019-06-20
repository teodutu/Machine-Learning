function [J, grad] = costFunction(theta, X, y)
  % COSTFUNCTION Compute cost and gradient for logistic regression
  % J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  % parameter for logistic regression and the gradient of the cost
  % w.r.t. to the parameters.

  m = length(y); % number of training examples

  % In similarity to the previous homework assignment, define a variable
  % holds the result of the sigmoid function so that it's called only once
  h = sigmoid(X * theta);

  % Following the new definition, compute J
  % The gradient remains almost the same as in the previous homework
  J = -1 * (y' * log(h) + (1 - y)' * log(1 - h)) / m;
  grad = X' * (h - y) / m;
end
