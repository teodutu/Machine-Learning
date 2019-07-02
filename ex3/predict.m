function p = predict(Theta1, Theta2, X)
  % PREDICT Predict the label of an input given a trained neural network
  % p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  % trained weights of a neural network (Theta1, Theta2)

  % Useful values
  m = size(X, 1);

  % Add ones to the X data matrix
  X = [ones(m, 1) X];

  % Calculate a2
  a2 = [ones(1, m); sigmoid(Theta1 * X')];

  % Calculate all prediction probabilities
  a3 = sigmoid(Theta2 * a2)';
  
  % Get the index of the maximum probability for each training example
  [p p] = max(a3, [], 2);
end
