function centroids = computeCentroids(X, idx, K)
  % COMPUTECENTROIDS returns the new centroids by computing the means of the 
  % data points assigned to each centroid.
  % centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
  % computing the means of the data points assigned to each centroid. It is
  % given a dataset X where each row is a single data point, a vector
  % idx of centroid assignments (i.e. each entry in range [1..K]) for each
  % example, and K, the number of centroids. You should return a matrix
  % centroids, where each row of centroids is the mean of the data points
  % assigned to it.

  % Calculate the distance to each centroid
  for i = 1:K
    % Get all indexes of features that are assign to centroid i 
    idx_i = find(idx == i);

    % Update each centroid by calculating the mean of the features assigned to it
    centroids(i, :) = sum(X(idx_i, :)) / length(idx_i);
  end
end

