function [bestEpsilon bestF1] = selectThreshold(yval, pval)
  % SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
  % outliers
  % [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
  % threshold to use for selecting outliers based on the results from a
  % validation set (pval) and the ground truth (yval).

  bestF1 = 0;
  stepsize = (max(pval) - min(pval)) / 1000;

  % Iterate through possible epsilons
  for epsilon = min(pval):stepsize:max(pval)
    % Get the actual anomaly prediction
    pred = pval < epsilon;

    % Count the true and false positives, as well as the false negatives
    truePos = sum(pred == 1 & yval == 1);
    falsePos = sum(pred == 1 & yval == 0);
    falseNeg = sum(pred == 0 & yval == 1);

    % Calculate the precision and recall
    prec = truePos / (truePos+falsePos);
    rec = truePos / (truePos+falseNeg);

    % Now get the F1 score
    F1 = 2*prec*rec / (prec+rec);

    % Update epsilon
    if F1 > bestF1
     bestF1 = F1;
     bestEpsilon = epsilon;
    end
  end
end
