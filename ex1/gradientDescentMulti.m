function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  % GRADIENTDESCENTMULTI Performs gradient descent to learn theta
  % theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
  % taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  [n n] = size(X); % the number of features

  for iter = 1:num_iters
    % Same as the univariate case
    S = X * theta - y;
    theta -= alpha * X' * S / m;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
  end
end
