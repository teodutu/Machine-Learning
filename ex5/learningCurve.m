function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
  % LEARNINGCURVE Generates the train and cross validation set errors needed 
  % to plot a learning curve
  % [error_train, error_val] = ...
  %     LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
  %     cross validation set errors for a learning curve. In particular, 
  %     it returns two vectors of the same length - error_train and 
  %     error_val. Then, error_train(i) contains the training error for
  %     i examples (and similarly for error_val(i)).
  %
  % In this function, you will compute the train and test errors for
  % dataset sizes from 1 up to m. In practice, when working with larger
  % datasets, you might want to do this in larger intervals.

  m = size(X, 1);  % number of training examples
  mval = size(Xval, 1);  % number of cross-validation tests

  for i = 1:m
    % Train for the first i samples
    theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
    
    % Compute the training error for these i samples 
    error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
    % Now, calculate the cross-validation error for the same first i trainint
    % examples
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
  end
end
