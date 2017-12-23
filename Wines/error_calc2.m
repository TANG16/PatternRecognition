function [n_testing_err, n_testing_succ, classif_for_testing] = error_calc2(wine_training, wine_testing, k, distance, dpar)
%dim 1: tr/te, 2: class, 3-15: real features
%Try classification
if strcmp(distance, 'neuclidean')
    mdl = fitcknn(normr(wine_training(:,3:end)), wine_training(:,2), 'NumNeighbors',k,'Distance', 'euclidean');
else
    mdl = fitcknn(wine_training(:,3:end), wine_training(:,2), 'NumNeighbors',k,'Distance', distance);
end
if strcmp(distance, 'minkowski') || strcmp(distance, 'mahalanobis') || strcmp(distance, 'seuclidean')
    mdl.DistParameter = dpar;
end
% classif_for_training = predict(mdl,wine_training(:,3:end));
if strcmp(distance, 'neuclidean')
    classif_for_testing = predict(mdl,normr(wine_testing(:,3:end)));
else
    classif_for_testing = predict(mdl,wine_testing(:,3:end));
end

%Error for training data (should be zero)
% error_for_training = classif_for_training - wine_training(:,2);
% n_training_err = nnz(error_for_training); %number of failures
% n_training_succ = size(error_for_training,1) - n_training_err; %number of successes
%Error for testing data (interesting to rate effectiveness)
error_for_testing = classif_for_testing - wine_testing(:,2);
n_testing_err = nnz(error_for_testing); %number of failures
n_testing_succ = size(error_for_testing,1) - n_testing_err; %number of successes

end

