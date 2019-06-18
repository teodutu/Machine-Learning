function J = computeCost(X, y, theta)
  % COMPUTECOST Compute cost for linear regression
  % J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
  % parameter for linear regression to fit the data points in X and y

  m = length(y); % number of training examples

  S = X * theta - y; % S stores each term of the sum
  J = S' * S / (2 * m); % the required sum of squared terms is equivalent to S' * S
end
