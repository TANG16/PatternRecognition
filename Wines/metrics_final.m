close all;
clear;
clc

%% Read data and define metrics

wine = dlmread('wine.data.csv');
numberofdata = size(wine,1); %178
dimensions = size(wine,2); %15 (1+1+13features)
classes = 3;
rng(7); 

metric_types = {'euclidean', 'cityblock', 'cosine', 'correlation', 'mink_{0.7}', 'mink_1', 'mink_2',...
                'mink_3', 'mink_4', 'mink_{100}', 'chebychev','seucl', ...
                'mahalanobis', 'mah_1', 'mah_2', 'mah_3', 'spearman', ...
                'chisquare', 'jensen-sh', 'earthmovers', 'neuclidean'}; 

metric_number = size(metric_types,2);

%% Split to training & testing data
wine_training = wine(wine(:,1)==1,:);
wine_testing = wine(wine(:,1)==2,:);

meansub = wine_training(:,3:end)-mean(wine_training(:,3:end),1);
covarmat = 1/size(wine_training, 1)*(meansub'*meansub);
class1 = wine_training(wine_training(:,2)==1,3:end)-mean(wine_training(wine_training(:,2)==1,3:end),1);
class2 = wine_training(wine_training(:,2)==2,3:end)-mean(wine_training(wine_training(:,2)==2,3:end),1);
class3 = wine_training(wine_training(:,2)==3,3:end)-mean(wine_training(wine_training(:,2)==3,3:end),1);
covarmat_class1 = 1/size(class1,1) * (class1'*class1);
covarmat_class2 = 1/size(class2,1) * (class2'*class2);
covarmat_class3 = 1/size(class3,1) * (class3'*class3);
standard_dev_vec = std(wine_training(:,3:end), 1);

%% Visualize difference between distance metrics
differences = zeros(numberofdata, numberofdata, metric_number); 
for i=1:metric_number
    [distance, dpar] = getMetricType_final(i, metric_types, covarmat, standard_dev_vec, covarmat_class1, covarmat_class2, covarmat_class3);
    X = wine;
    if strcmp(distance, 'minkowski') || strcmp(distance, 'mahalanobis') || strcmp(distance, 'seuclidean')
        Z = squareform(pdist(X(:,3:end),distance,dpar));
    elseif strcmp(distance, 'neuclidean')
        Z = squareform(pdist(normr(X(:,3:end)),'euclidean'));   
    else
        Z = squareform(pdist(X(:,3:end),distance));
    end
    differences(:,:,i) = Z; 
end
figure
hold on
b1 = reshape(differences(1,2,:),1,size(differences,3));
b2 = reshape(differences(1,3,:),1,size(differences,3));
b100 = reshape(differences(1,100,:),1,size(differences,3));
b = bar3(1:metric_number,[b100',b2',b1']);
title('Distance from different metric types');
set(gca,'yticklabel',metric_types,'YTick',1:numel(metric_types));
ax = gca;
ax.YTickLabelRotation = 45; 
ax.XTickLabelRotation = -45; 
grid on
l = cell(1,3);
l{1}='diff(data1,data2)'; l{2}='diff(data1,data3)'; l{3}='diff(data1,data100)';
legend(b,l);
view(-70,40);


%% Distance metrics / NN-classification (k=1 based on lectures)
temp = zeros(metric_number,4);
time_knn = zeros(1,metric_number);
pred_maha_saved = -1*ones(size(wine_testing,1),4);
for i=1:metric_number
    [distance, dpar] = getMetricType_final(i, metric_types, covarmat, standard_dev_vec, covarmat_class1, covarmat_class2, covarmat_class3);
    tic
    [temp(i,3), ~, pred1] = error_calc2(wine_training, wine_testing, 1, distance, dpar);
    time_knn(i) = toc;
    
    if isequal(dpar, covarmat)
        pred_maha_saved(:,1) = pred1;
    elseif isequal(dpar, covarmat_class1)
        pred_maha_saved(:,2) = pred1;
    elseif isequal(dpar, covarmat_class2)
        pred_maha_saved(:,3) = pred1;
    elseif isequal(dpar, covarmat_class3)
        pred_maha_saved(:,4) = pred1;
    end
end

%Interesting, but nothing intuitive, not sure if we should include this
%result
class1_acc = sum(pred_maha_saved(wine_testing(:,2)==1, :) == 1);
class2_acc = sum(pred_maha_saved(wine_testing(:,2)==2, :) == 2);
class3_acc = sum(pred_maha_saved(wine_testing(:,2)==3, :) == 3);
    
    
figure
bar3(100*temp(:,3)./size(wine_testing,1)); %plot error rate %
grid on
title('kNN classification error (k=1)');
set(gca,'Yticklabel',metric_types,'YTick',1:numel(metric_types));
ax = gca;
ax.YTickLabelRotation = 65; 
view(-70,30);
saveas(gcf,['knn_acc.jpg']);

figure
bar3(time_knn);
grid on
title('kNN time consumption (k=1) [s]');
set(gca,'Yticklabel',metric_types,'YTick',1:numel(metric_types));
ax = gca;
ax.YTickLabelRotation = 65; 
view(-70,30);
saveas(gcf,['knn_time.jpg']);

%% Earth-movers distance 
% How it changes if we shuffle the order of the features
[distance, dpar] = getMetricType_final(getIndex('earthmovers', metric_types), metric_types, covarmat, standard_dev_vec, covarmat_class1, covarmat_class2, covarmat_class3);
eclass = zeros(1,40);
[eclass(1), ~, pred1] = error_calc2(wine_training, wine_testing, 1, distance, dpar);

wine_end = wine_training(:,3:end);
wine_training_shuffledbins = zeros(size(wine_training));
oldorder = 1:13;
for i=2:40
    order = randperm(size(wine_end,2));
    while nnz(order - oldorder) == 0
        i;
            order = randperm(size(wine_end,2));
    end
    wine_training_shuffledbins = [wine_training(:,1:2) wine_end(:, order) ];
    [eclass(i), ~, pred2] = error_calc2(wine_training_shuffledbins, wine_testing, 1, distance, dpar);
    oldorder = order;
end

figure
bar(100*eclass./size(wine_testing,1));
grid on
title('Earth movers distance (shuffled bins)');
ylabel('Classification error (%)');

%% Exact kMkNN search

%Build-up stage
rng(7);
kc = floor(sqrt(size(wine_training,1))); 
tt = randperm(size(wine_training,1));
x = tt(1:kc);
initial_means = wine_training(x,:);
pi = cell(kc);
di = cell(kc);
maxdistance = 10^5;
k = 1;
pred = zeros(size(wine_testing,1),1);
speed = zeros(1,metric_number);
error = zeros(1,metric_number);
for mn=1:metric_number
    [distance, dpar] = getMetricType_final(mn, metric_types, covarmat, standard_dev_vec, covarmat_class1, covarmat_class2, covarmat_class3);
    [class_kmknn_vec, mean_kmknn_vec] =  simplekmeans(wine_training(:,3:end), initial_means(:,3:end),100, distance, dpar);
    for class=1:kc
        temp = wine_training(class_kmknn_vec==class,:);
        if strcmp(distance, 'minkowski') || strcmp(distance, 'mahalanobis') || strcmp(distance, 'seuclidean')
            di_temp = pdist2(temp(:,3:end), mean_kmknn_vec(class,:), distance, dpar);
        elseif strcmp(distance, 'neuclidean')
            di_temp = pdist2(normr(temp(:,3:end)), normr(mean_kmknn_vec(class,:)), 'euclidean');
        else
            di_temp = pdist2(temp(:,3:end), mean_kmknn_vec(class,:), distance);
        end
        [di_temp,index] = sort(di_temp, 'descend');
        di{class} = di_temp;
        pi{class} = temp(index,:);
    end
%Search stage
    tic
    for ts = 1:size(wine_testing,1)
        object = maxdistance*ones(k,2); %first col is actual distance second is the class
        test_sample = wine_testing(ts,3:end);
        if strcmp(distance, 'minkowski') || strcmp(distance, 'mahalanobis') || strcmp(distance, 'seuclidean')
            dist_to_centres = pdist2(test_sample, mean_kmknn_vec, distance, dpar);
        elseif strcmp(distance, 'neuclidean')
            dist_to_centres = pdist2(normr(test_sample), normr(mean_kmknn_vec), 'euclidean');
        else
            dist_to_centres = pdist2(test_sample, mean_kmknn_vec, distance);
        end
        [~, ix] = sort(dist_to_centres, 'ascend');
        for cluster = ix
            for member = 1:length(di{cluster})
                pc = di{cluster}(member);
                compare = dist_to_centres(cluster)-pc; %If abs taken, does not work!
                if max(object(:,1)) <= compare
%                     disp('Some training samples skipped due to triangular conidition')
%                     disp(['ts:', num2str(ts)])
%                     disp(['cluster:', num2str(cluster)])
%                     disp(['member:', num2str(member)])
                    break
                else
                    if strcmp(distance, 'minkowski') || strcmp(distance, 'mahalanobis') || strcmp(distance, 'seuclidean')
                        temp = pdist2(test_sample, pi{cluster}(member,3:end), distance, dpar);
                    elseif strcmp(distance, 'neuclidean')
                        temp = pdist2(normr(test_sample), normr(pi{cluster}(member,3:end)), 'euclidean');
                    else
                        temp = pdist2(test_sample, pi{cluster}(member,3:end), distance);
                    end
                    if temp < max(object(:,1))
                        [~,idx] = max(object(:,1));
                        object(idx,:) = [temp, pi{cluster}(member,2)];
                    end
                end
            end
        end
        pred(ts) = mode(object(:,2));
        % This is to be used is k>1 and majority voting (mode) cannot decide
        %     [~, ~, C] = mode(object(:,2));
        %     C = cell2mat(C);
        %     MV_dist = zeros(1,length(c)); %distances for majority voted classes
        %     for j = 1:length(C)
        %         class = C(j);
        %         MV_dist(j) = mean(object(object(:,2) == class, 1));
        %     end
        %     [~, pred(ts)] = min(MV_dist);
    end
    speed(mn) = toc;
    error(mn) = nnz(pred-wine_testing(:,2));
end
figure
bar3(100*error./size(wine_testing,1)); %plot error rate %
grid on
title('kMkNN classification (k=1) with different metrics');
set(gca,'Yticklabel',metric_types,'YTick',1:numel(metric_types));
%ylabel('Classification error (%)');
ax = gca;
ax.YTickLabelRotation = 65; 
view(-70,30);
%saveas(gcf,['kmknn_acc.jpg']);
figure
bar3(speed);
title('kMkNN time consumption (k=1) [s]');
%set(gca,'xticklabel',metric_types,'XTick',1:numel(metric_types));
%ylabel('Time consumption [s]');
set(gca,'Yticklabel',metric_types,'YTick',1:numel(metric_types));
ax = gca;
ax.YTickLabelRotation = 65; 
view(-70,30);
saveas(gcf,['kmknn_time.jpg']);
              
        
    
    