function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
  % VALIDATIONCURVE Generate the train and validation errors needed to
  % plot a validation curve that we can use to select lambda
  % [lambda_vec, error_train, error_val] = ...
  %     VALIDATIONCURVE(X, y, Xval, yval) returns the train
  %     and validation errors (in error_train, error_val)
  %     for different values of lambda. You are given the training set (X,
  %     y) and validation set (Xval, yval).

  % Select values for lambda
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
  l = length(lambda_vec);

  for i = 1:l
    % Train for the classifier for the current lambda
    theta = trainLinearReg(X, y, lambda_vec(i));
    
    % Compute the training error for this lambda
    error_train(i) = linearRegCostFunction(X, y, theta, 0);
    % Now, calculate the cross-validation error for the same lambda
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
  end
end
