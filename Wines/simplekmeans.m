function [class_vec, mean_vec] = simplekmeans(vec, initial_means, maxiterkmeans, distance, dpar)
%Class 1 is first row vector in initial_means

mean_vec = initial_means;
mean_minus = zeros(size(initial_means));
index = 0;
K = size(initial_means, 1); %2 in test data, 3 in wine
while ~isequal(mean_minus, mean_vec)
    mean_minus = mean_vec;
%    res_matrix = bsxfun(@plus, sum(vec.^2, 2), bsxfun(@minus, (sum(mean_vec.^2, 2))', 2*vec*mean_vec')); %I want (SAMPLEi - MUx)^2, I calculate SAMPLEi^2+MUx^2-2*SAMPLEi*MUx
    %X = [vec; mean_vec]; Use pdist2, more efficient, no unneccessary
    %distance calcs!
    if strcmp(distance, 'minkowski') || strcmp(distance, 'mahalanobis') || strcmp(distance, 'seuclidean')
        res_matrix = pdist2(vec,initial_means,distance,dpar);
    elseif strcmp(distance, 'neuclidean')
        res_matrix = pdist2(normr(vec), normr(initial_means), 'euclidean');
    else
        res_matrix = pdist2(vec,initial_means,distance);
    end
%     Z(1:size(X,1)-K,end-K+1:end);
%     res_matrix = Z(1:size(X,1)-K,end-K+1:end);

    [~, ix] = min(res_matrix, [], 2);
    class_vec = ix;
    for class = 1:K
        mean_vec(class,:) = mean(vec(class_vec==class, :), 1);
    end
    index = index + 1;
    if index > maxiterkmeans
        break
    end
    %[mean_minus mean_vec] %NaN was caused by centroids being the same
    %because multiple same entries in original dataset
    %pause
end
%index;
end

% To check exercise results in 2010 exam
% 
% init_mean = [1 0 0; 0 1 0]
% vec = [1 3 4; 4 2 5; 2 5 6; 1 3 3; 2 0 4; 4 1 3]
% [class_vec, mean_vec] = simplekmeans(vec, init_mean)

