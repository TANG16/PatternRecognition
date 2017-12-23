%% Pattern Recognition Coursework Question 2-2

clear 
close all
clc
load('face.mat')

height = 56;
width = 46;
number = size(X,2);
images = zeros(height,width,number);
persons = 52; %52 class, 10 pictures of each person

for i=1:number
    im = reshape(X(:,i),height,width);
    images(:,:,i) = im;
end

%% Randomized samples within each class
%Uncomment this part to shuffle the images

% images_rand = zeros(height,width,number);
% set seed for reproducibility
% rng(7);
% for i = 1:persons
%     start = 10*(i-1)+1;
%     section = images(:,:,start:start+9);
%     section_rand = section(:,:,randperm(size(section,3)));
%     images_rand(:,:,start:start+9) = section_rand;
% end
% images = images_rand; %Uncomment this to shuffle the images

%% Divide set into training and testing data set
%Every following 10 images belong to one class
%Use the first n image of every class for training
n = 7; %70-30 train-test split

trainimages = zeros(height, width);
testimages = zeros(height, width);
ltrain = [1];
ltest = [1];
for i=1:persons
    firsttrain = (i-1)*10 + 1;
    trainimages = cat(3, trainimages, images(:,:,firsttrain:(firsttrain+n-1))); %1:7, 11:17, ...
    testimages = cat(3, testimages, images(:,:,(firsttrain+n):(firsttrain+9))); %8:10, 18:20, ...
    ltrain = cat(2,ltrain,l(firsttrain:firsttrain+n-1));
    ltest = cat(2,ltest,l((firsttrain+n):(firsttrain+9)));
end
trainimages = trainimages(:,:,2:end); %remove first zero matrix
testimages = testimages(:,:,2:end); %remove first zero matrix
ltest = ltest(2:end);
ltrain = ltrain(2:end); %remove first zero

%% Training class means
Xtrain = reshape(trainimages, 56*46, 364); %NOT EQUAL TO reshape(trainimages,[56*46, 364])' since reshape goes down each column, versus transpose goes in each row!!
Xtest = reshape(testimages, 56*46, 156);
Xtrain = Xtrain';
Xtest = Xtest';
Xtrain_class_avg = reshape(Xtrain, 7,52,56*46);
Xtrain_class_avg = mean(Xtrain_class_avg, 1);
Xtrain_class_avg = reshape(Xtrain_class_avg, 52,56*46);

%% Between class scatter
global_mean = mean(Xtrain, 1);
temp = bsxfun(@minus, Xtrain_class_avg, global_mean);
between_class_scatter = temp'*temp;

%% Within class scatter
within_class_scatter = zeros(56*46,56*46);
for class = 1:52
    temp = bsxfun(@minus, Xtrain(7*class-6:7*class, :), Xtrain_class_avg(class,:));
    within_class_scatter = within_class_scatter + temp'*temp;
end

%% PCA (M_PCA can be between c-1 and N-c, so between 51 and 7*52-52, so between 51 and 312, indeed confirmed, rank(within_class_scatter) = 312, rank(between_class_scatter=51))
% N is the number of training samples (364)
% c is th number is classes (52)
% M_PCA is the number of eigenvectors used for PCA
Z = bsxfun(@minus, Xtrain, global_mean);
covariance_matrix = 1/size(Xtrain,1) *(Z'*Z);
[eigvec, eigval] = eig(covariance_matrix); %checked, same as Rhat, smallest front, so flip
eigval = diag(eigval);
eigval = eigval(end:-1:1); %sort eigenvalues in descending order
eigvec = fliplr(eigvec); %sort eigenvectors in descending order

%% PCA cont. & LDA & Nearest Neighbour settings and implementation, M_LDA <= 51
% M_LDA is the number of generalized eigenvectors used for LDA
index = 1;
M_PCA_vec = [51:50:312 312];
M_LDA_vec = [3:20:43, 51];
neighbours_vec = [1 2 3 4 5];
accuracy = zeros(length(M_PCA_vec),length(M_LDA_vec),length(neighbours_vec));
for i = 1:length(M_PCA_vec)
    for j = 1:length(M_LDA_vec)
        for k = 1:length(neighbours_vec)
            M_PCA = M_PCA_vec(i);
            M_LDA = M_LDA_vec(j);
            neighbours = neighbours_vec(k);
            eigvec_i = eigvec(:,1:M_PCA);
            eigval_i = eigval(1:M_PCA);

            %projections
            Xtrain_projected = Xtrain*eigvec_i;
            BCS_PCA = eigvec_i'*between_class_scatter*eigvec_i;
            WCS_PCA = eigvec_i'*within_class_scatter*eigvec_i;

            %projection for test set
            Xtest_projected = Xtest*eigvec_i;

            % LDA, solve the generalized eigenvalue problem, 
            [generalized_eigvec, generalized_eigval] = eig(BCS_PCA, WCS_PCA); %already decreasing order

            %M_LDA = 51; 
            generalized_eigvec_i = generalized_eigvec(:,1:M_LDA);
            generalized_eigval_i = diag(generalized_eigval(1:M_LDA,1:M_LDA));
            %disp(generalized_eigval);

            %projection for training set
            Xtrain_LDA = Xtrain_projected*generalized_eigvec_i;

            %projection for test set
            Xtest_LDA = Xtest_projected*generalized_eigvec_i;
            
            %nearest neighbour classification
            classknn = knnclassify(Xtest_LDA, Xtrain_LDA, ltrain', neighbours);
            res = sum((classknn==ltest')) / length(ltest);
            if res > max(accuracy(:))
                best_prediction = classknn;
            end
            accuracy(i,j,k) = res;
            index = index + 1;
            disp(index);
        end
    end
end

%% Best result and its confusion matrix
%Best parameters
[val, idx] = max(accuracy(:));
[best_M_PCA_idx, best_M_LDA_idx, best_neighbours_idx] = ind2sub(size(accuracy), idx);

%Confusion matrix
res_matrix = [ltest', best_prediction];
confusion = zeros(52,52);
for i = 1:size(res_matrix,1)
    y = res_matrix(i,1);
    x = res_matrix(i,2);
    confusion(y,x) = confusion(y,x) + 1;
end
confusion = confusion/3; %3 test images in each class
acc = trace(confusion)/52; %52 classes
imagesc(confusion); colorbar
title(['Confusion matrix for PCA-LDA-NN classification using the best parameters, Accuracy = ', num2str(acc)], 'FontSize', 20)
xlabel('Predicted classes', 'FontSize', 16)
ylabel('Actual classes', 'FontSize', 16)

%% Committee Machine - Bootstrapping
no_models = 200; %number of base models 
idx_increment = 7*(ones(no_models,1)*(ltrain-1))';
rng(7)
bootstrap_mat = ceil(7*rand(7*52,no_models))+ idx_increment;
%7 randomized samples, 52 classes, 10 models

%% Committee Machine - Creation of random subspace and nearest neighbour classification of ensemble models
%See reference is report
%scatter(30:400, eigval(30:400)) 
%From figure, we choose M0 to be 33/49/100 (see Page 18, ensemble learning slides)
M0 = 49; %randomness parameter
M_PCA = 151;
M1 = M_PCA-M0; 
remain = 363-M0; %363 is N(=52-7)-1

%commonly chosen subspace
common_eigvec = eigvec(:,1:M0);

%scatter(1:51, generalized_eigval_i)
%From figure, we choose M0 to be 31/51
M_LDA = 51;

%initialize predicted classes and model_accuracies; 3*52 test samples, 10 models 
class_pred = zeros(3*52,no_models);
model_accuracies = zeros(1,no_models);

for model = 1:no_models
    
    %additionally chosen subspace
    rng(model);
    ACS = M0+randperm(remain, M1);
    additional_eigvec = eigvec(:,ACS);
    
    %complete model subspace
    model_eigvec = [common_eigvec,additional_eigvec];
    
    %model samples
    model_train = Xtrain(bootstrap_mat(:,model),:);
    
    %model class means
    train_class_avg = reshape(model_train, 7,52,56*46);
    train_class_avg = mean(train_class_avg, 1);
    train_class_avg = reshape(train_class_avg, 52,56*46);
    
    %between class scatter
    model_mean = mean(model_train, 1);
    temp = bsxfun(@minus, train_class_avg, model_mean);
    model_BCS = temp'*temp;
    
    %within class scatter
    model_WCS = zeros(56*46,56*46);
    for class = 1:52
        temp = bsxfun(@minus, model_train(7*class-6:7*class, :), train_class_avg(class,:));
        model_WCS = model_WCS + temp'*temp;
    end
    
    %projections
    model_train_projected = model_train*model_eigvec;
    model_BCS_PCA = model_eigvec'*model_BCS*model_eigvec;
    model_WCS_PCA = model_eigvec'*model_WCS*model_eigvec;
    
    %projection for test set
    model_test_projected = Xtest*model_eigvec;
    
    % LDA, solve the generalized eigenvalue problem,
    [model_generalized_eigvec, model_generalized_eigval] = eig(model_BCS_PCA, model_WCS_PCA); %already decreasing order
    
    model_generalized_eigvec = model_generalized_eigvec(:,1:M_LDA);
    model_generalized_eigval = diag(model_generalized_eigval(1:M_LDA,1:M_LDA));
    
    %projection for training set
    model_train_LDA = model_train_projected*model_generalized_eigvec;
    
    %projection for test set
    model_test_LDA = model_test_projected*model_generalized_eigvec;
    
    %nearest neighbour classification
    neighbours = 3;
    class_pred(:,model) = knnclassify(model_test_LDA, model_train_LDA, ltrain', neighbours);
    model_accuracies(model) = sum((class_pred(:,model)==ltest')) / length(ltest);
    
end

%% Combine results from ensemble models
predicted_classes = mode(class_pred,2);
ensemble_accuracy = sum((predicted_classes==ltest')) / length(ltest);

%% Confusion matrix
res_matrix = [ltest', predicted_classes];
confusion = zeros(52,52);
for i = 1:size(res_matrix,1)
    y = res_matrix(i,1);
    x = res_matrix(i,2);
    confusion(y,x) = confusion(y,x) + 1;
end
confusion = confusion/3; %3 test images in each class
acc = trace(confusion)/52; %52 classes
imagesc(confusion); colorbar
title(['Confusion matrix for combined ensemble models of PCA-LDA-NN classification, Accuracy = ', num2str(acc)], 'FontSize', 20)
xlabel('Predicted classes', 'FontSize', 16)
ylabel('Actual classes', 'FontSize', 16)

%% Average error of the models
disp(mean(model_accuracies)) %0.3421 vs 0.8205



    

