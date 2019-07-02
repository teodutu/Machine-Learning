function [C, sigma] = dataset3Params(X, y, Xval, yval)
  % DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
  % where you select the optimal (C, sigma) learning parameters to use for SVM
  % with RBF kernel
  % [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
  % sigma. You should complete this function to return the optimal C and 
  % sigma based on a cross-validation set.

  % Initialise variables
  Cvec = sigmavec =  [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
  l = length(Cvec);
  bestCIdx = 0;
  bestSigmaIDx = 0;
  minError = 1e100;

  % Iterate through all the suggested Cs and sigmas and choose the pair that
  % minimises the error on the cross-validation set
  for i = 1:l
    for j = 1:l
      % Train the SVM for the current C and sigma
      model = svmTrain(X, y, Cvec(i), @(x1, x2) gaussianKernel(x1, x2, sigmavec(j)));

      % Test the SVM on the cross-validation set
      predictions = svmPredict(model, Xval);

      % Calculate the error
      error = mean(double(predictions ~= yval));

      % If the current error is smaller than the previous lowest error, then we
      % might have a minimum
      if minError > error
        minError = error;
        bestCIdx = i;
        bestSigmaIdx = j;
      end
    end
  end

  % Return the minimum values for C and sigma
  C = Cvec(bestCIdx);
  sigma = sigmavec(bestSigmaIdx);
end
