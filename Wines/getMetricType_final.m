function [ distance, dpar ] = getMetricType_final( i, metric_type, covarmat, standard_dev_vec, covarmatclass1, covarmatclass2, covarmatclass3)

dpar = 0;
    switch i
        case getIndex('euclidean', metric_type)
            distance = 'euclidean';
        case getIndex('cityblock', metric_type)
            distance = 'cityblock';
        case getIndex('cosine', metric_type)
            distance = 'cosine';
        case getIndex('correlation', metric_type)
            distance = 'correlation';
        case getIndex('neuclidean', metric_type)
            distance = 'neuclidean';      
        case getIndex('crosscorr', metric_type)
            crosscorr = @(x,Z) (x*Z')'; 
            distance = crosscorr;
        case getIndex('mink_{0.7}', metric_type)
            distance = 'minkowski';
            dpar = 0.7;
        case getIndex('mink_1', metric_type)
            distance = 'minkowski'; %default p=1, should equal to cityblock
            dpar = 1;
        case getIndex('mink_2', metric_type)
            distance = 'minkowski'; %default p=2, should equal to eucledian
            dpar = 2;
        case getIndex('mink_3', metric_type)
            distance = 'minkowski';
            dpar = 3;
        case getIndex('mink_4', metric_type)
            distance = 'minkowski';
            dpar = 4;
        case getIndex('mink_{100}', metric_type)
            distance = 'minkowski';
            dpar = 100;
        case getIndex('seucl', metric_type)
            distance = 'seuclidean';
            dpar = standard_dev_vec;
        case getIndex('chebychev', metric_type)
            distance = 'chebychev';
        case getIndex('jaccard', metric_type)
            distance = 'jaccard';
        case getIndex('mahalanobis', metric_type)
            distance = 'mahalanobis';
            dpar = covarmat;
        case getIndex('mah_1', metric_type)
            distance = 'mahalanobis';
            dpar = covarmatclass1;
        case getIndex('mah_2', metric_type)
            distance = 'mahalanobis';
            dpar = covarmatclass2;
        case getIndex('mah_3', metric_type)
            distance = 'mahalanobis';
            dpar = covarmatclass3;
        case getIndex('spearman', metric_type)
            distance = 'spearman';
        case getIndex('chisquare', metric_type)
            chisquare = @(x,Z)sqrt( sum((bsxfun(@minus,x,Z).^2) ./ bsxfun(@plus,x,Z), 2)/2 );
            distance = chisquare;
        case getIndex('jensen-sh', metric_type)
            jensenshannon = @(x,Z)sqrt( sum( x .* log((2*x)./ bsxfun(@plus,x,Z)), 2)/2 + sum( Z .* log((2*Z)./ bsxfun(@plus,x,Z)), 2)/2 );
%            jensenshannon = @(x,Z)sqrt( sum( x .* log10(2*x)./ bsxfun(@plus,x,Z), 2)/2 ) + sum( Z .* log10((2*Z)./ bsxfun(@plus,x,Z)), 2)/2 );
            distance = jensenshannon;
        case getIndex('earthmovers', metric_type)
%             %test with pdist([x; y], earthmovers) = 3.3
%             x = [0 0 0 0.2 0.3 0.5 0 0 0 0 0];
%             y = [0 0 0 0 0 0 0.1 0.2 0.7 0 0];
            earthmovers = @(x,Z)(  sum( abs(bsxfun(@minus, cumsum(x./sum(x,2),2), cumsum(Z./sum(Z,2),2))), 2));
            distance = earthmovers;
%         case 
%             distance = ;
        otherwise
            warning('Undefined distance metric used')
    end

end

