function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  % GRADIENTDESCENT Performs gradient descent to learn theta
  % theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
  % taking num_iters gradient steps with learning rate alpha

  m = length(y); % number of training examples

  % Can be optimised by adding a tolerance `tol` and iterating as long as
  % the difference between the last 2 Js is smaller than said `tol`
  for iter = 1:num_iters
    S = X * theta - y; % S has the same significance as in `computeCost.m`

    % theta0 and theta1 must be updated independently
    % This is achieved by computing `S` before updating any theta
    % By multiplying `X'` and `S` we obtain a vector that contains the required
    % sums
    theta -= alpha * X' * S / m;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
  end
end
