function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
  % COFICOSTFUNC Collaborative filtering cost function
  % [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  % num_features, lambda) returns the cost and gradient for the
  % collaborative filtering problem.

  % Unfold the X and Theta matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);     

  % First compute the matrix of the prediction error
  Err = X * Theta' - Y;

  % Calculate J and the gradients in unregularised form
  J_unreg = sum(sum(Err.^2.*R)) / 2;
  X_grad_unreg = (Err.*R) * Theta;
  Theta_grad_unreg = (Err.*R)' * X;

  % Now regularise all of the above
  J = J_unreg + lambda*(sum(sum(Theta.^2)) + sum(sum(X.^2)))/2;
  Theta_grad = Theta_grad_unreg + lambda * Theta;
  X_grad = X_grad_unreg + lambda * X;

  % Liniarise the gradients into a single column vector
  grad = [X_grad(:); Theta_grad(:)];
end