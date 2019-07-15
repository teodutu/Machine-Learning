function idx = findClosestCentroids(X, centroids)
  % FINDCLOSESTCENTROIDS computes the centroid memberships for every example
  % idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
  % in idx for a dataset X where each row is a single example. idx = m x 1 
  % vector of centroid assignments (i.e. each entry in range [1..K])

  % Set K and m
  K = size(centroids, 1);
  m = size(X,1);

  % Process each index independently
  for i = 1:m
    % Calculate the distance to each centroid
    for j = 1:K
      dist(j) = sqrt(sum((X(i, :) - centroids(j, :)).^2));
    end

    % Get the minimum of the previously calculated indexes
    [~, idx(i)] = min(dist);
  end
end