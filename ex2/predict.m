function p = predict(theta, X)
  % PREDICT Predict whether the label is 0 or 1 using learned logistic 
  % regression parameters theta
  % p = PREDICT(theta, X) computes the predictions for X using a 
  % threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

  % The prediction means comparing each elemnt of X * theta to 0
  p = (X * theta) >= 0;
end
