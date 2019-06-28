function [J grad] = nnCostFunction(nn_params,
                                   input_layer_size,
                                   hidden_layer_size,
                                   num_labels,
                                   X, y, lambda)
  % NNCOSTFUNCTION Implements the neural network cost function for a two layer
  % neural network which performs classification
  % [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  % X, y, lambda) computes the cost and gradient of the neural network. The
  % parameters for the neural network are "unrolled" into the vector
  % nn_params and need to be converted back into the weight matrices. 
  % 
  % The returned parameter grad should be a "unrolled" vector of the
  % partial derivatives of the neural network.
  %

  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),
                   num_labels, (hidden_layer_size + 1));

  [m n] = size(X); % number of training examples and that of features

  % Add ones to the X data matrix
  X = [ones(m, 1) X];                 

  % Calculate a2
  z2 = Theta1 * X';
  a2 = [ones(1, m); sigmoid(z2)];

  % Calculate all prediction probabilities
  z3 = Theta2 * a2;
  a3 = sigmoid(z3)';

  % Make the output matrix Y  
  Y = (1:num_labels) == y;

  % Calculate J in an unregularised form, then regularise it
  J_unreg = -1 * sum(sum(Y .* log(a3) + (1 - Y) .* log(1 - a3))) / m;
  regFactor = lambda * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))) / (2*m);
  J = J_unreg + regFactor;

  % Calculate Theta2_grad
  delta3 = a3 - Y;
  Theta2_grad = (delta3' * a2' + lambda * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]) / m ;

  % Calculate Theta1_grad
  delta2 = (delta3 * Theta2) .* [ones(1, m); sigmoidGradient(z2)]';
  delta2 = delta2(:, 2:end);
  Theta1_grad = (delta2' * X + lambda * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]) / m;

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
