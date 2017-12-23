%% Pattern Recognition Coursework Question 1

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

%% Plot images
%close all
%view all images of the first 10 persons (10 classes)
for a=1:10
    figure
    for i=1:10 %belonging to one class
        subplot(3,4,i); imshow(images(:,:,((a-1)*10) + i),[]); % 1:10; 11:21, 21:31, ...
    end
end
% for a=1:10%persons
%     figure
%     for i=1:10 %belonging to one class
%         subplot(3,4,i); imshow(images_rand(:,:,((a-1)*10) + i),[]); % 1:10; 11:21, 21:31, ...
%     end
% end

%% Plot training images
close all
for j=1:persons 
    figure
    for i=1:n %belonging to one class
        subplot(3,4,i); imshow(trainimages(:,:,(j-1)*n+i),[]);
    end
end

%% Plot testing images
close all
for k=1:persons
    figure
    for i=1:(10-n) %belonging to one class
        subplot(3,4,i); imshow(testimages(:,:,(k-1)*(10-n) + i),[]);
    end
end

%% Calculate the mean in training data set
 im_mean = mean(trainimages,3);
 figure; imshow(uint8(im_mean));
 
 %% Normalize faces by mean substraction & Calculate covariance matrix
 Xtrain = reshape(trainimages,[size(X,1),size(trainimages,3)]); %Training data in vector form
 Xtest = reshape(testimages,[size(X,1),size(testimages,3)]); %Testing data in vector form
 mu_hat = im_mean(:)*ones(1,size(Xtrain,2)); %same mean vector in all columns for all images
 %Note: could use bsxfun 
 Z = Xtrain - mu_hat;
 R_hat = (Z*Z')/size(Xtrain,2); %we divide by the no. of samples which is size(Xtrain,2)
 
 %% PCA from S=(1/N)AA'
[eig_vec, eig_val] = eig(R_hat);
eigval = diag(eig_val);
eigval = eigval(end:-1:1); %sort eigenvalues in descending order
eigvec = fliplr(eig_vec); %sort eigenvectors in descending order

%% Plot first t eigenfaces
close all;
t = 50;
figure (1)
title('First 50 eigenfaces', 'FontSize', 20)
for i=1:t
    subplot(5,10,i);
    imshow(reshape(eigvec(:,i), height, width), []);
end

%% PCA from S=(1/N)A'A
%rank of covariance matrix is N-1 = 363
%Same eigenvectors (1st 363 biggest) after normalization (or in some cases
%times -1, but that is still the same vector)
R_hat2 = (Z'*Z)/size(Xtrain,2); 
[eig_vec2, eig_val2] = eig(R_hat2);
eigval2 = diag(eig_val2);
%eigval2 = eigval2(end:-1:1); %sort eigenvalues in descending order
[eigval2, ix] = sort(eigval2, 'descend');
eigvec2 = eig_vec2(:,ix); %sort eigenvectors in descending order
eigvec2 = Z*eigvec2; %highest eigenvectors of AA' Page 17

%% Checking if each column in eigvec and eigvec2 are the same
% result already normalized, check, for example, sum(eigvec(:,50).^2)
scale = eigvec2./(eigvec(:,1:364));
sum(diff(scale)) %very very small, numerical error

%% Calculate projections
%Check training and testing images both
projection_training = Z'*eigvec; %one row contains proj coordinates for one image
projection_testing = (Xtest - mu_hat(:,1:size(Xtest,2)))'*eigvec; %substract training mean

t = 300;
reconstructed_training = eigvec(:,1:t)*projection_training(:,1:t)' + im_mean(:)*ones(1,size(Xtrain,2));
reconstructed_testing = eigvec(:,1:t)*projection_testing(:,1:t)' + im_mean(:)*ones(1,size(Xtest,2));

figure

%training samples
subplot(4,6,1); imshow(trainimages(:,:,1),[]);
subplot(4,6,2); imshow(trainimages(:,:,8),[]);
subplot(4,6,3); imshow(trainimages(:,:,15),[]);
subplot(4,6,4); imshow(trainimages(:,:,22),[]);
subplot(4,6,5); imshow(trainimages(:,:,29),[]);
subplot(4,6,6); imshow(trainimages(:,:,36),[]);

subplot(4,6,7); imshow(reshape(reconstructed_training(:,1), height, width), []);
subplot(4,6,8); imshow(reshape(reconstructed_training(:,8), height, width), []);
subplot(4,6,9); imshow(reshape(reconstructed_training(:,15), height, width), []);
subplot(4,6,10); imshow(reshape(reconstructed_training(:,22), height, width), []);
subplot(4,6,11); imshow(reshape(reconstructed_training(:,29), height, width), []);
subplot(4,6,12); imshow(reshape(reconstructed_training(:,36), height, width), []);

%testing
subplot(4,6,13); imshow(testimages(:,:,1),[]);
subplot(4,6,14); imshow(testimages(:,:,4),[]);
subplot(4,6,15); imshow(testimages(:,:,7),[]);
subplot(4,6,16); imshow(testimages(:,:,10),[]);
subplot(4,6,17); imshow(testimages(:,:,13),[]);
subplot(4,6,18); imshow(testimages(:,:,16),[]);

subplot(4,6,19); imshow(reshape(reconstructed_testing(:,1), height, width), []);
subplot(4,6,20); imshow(reshape(reconstructed_testing(:,4), height, width), []);
subplot(4,6,21); imshow(reshape(reconstructed_testing(:,7), height, width), []);
subplot(4,6,22); imshow(reshape(reconstructed_testing(:,10), height, width), []);
subplot(4,6,23); imshow(reshape(reconstructed_testing(:,13), height, width), []);
subplot(4,6,24); imshow(reshape(reconstructed_testing(:,16), height, width), []);

%% Reconstruction error example, t = 100

t = 100;

reconstructed_training = eigvec(:,1:t)*projection_training(:,1:t)' + im_mean(:)*ones(1,size(Xtrain,2));
reconstructed_testing = eigvec(:,1:t)*projection_testing(:,1:t)' + im_mean(:)*ones(1,size(Xtest,2));

rec_error_training = 1/size(Xtrain,2)*sum(sum((Xtrain - reconstructed_training).^2,1)); %2.5773e+05 for t = 100
rec_error_testing = 1/size(Xtest,2)*sum(sum((Xtest - reconstructed_testing).^2,1));

% Check theory
sum(eigval(t+1:end)) % 2.5773e+05, indeed the same

%PCA Method Two (see Report)
projection_training_methodtwo = Xtrain'*eigvec; %second method, original picture is projected and not the one whch had its mean subtracted!
reconstructed_training_methodtwo = eigvec(:,1:t)*projection_training_methodtwo(:,1:t)'; %second method, in this case do not add the means back
unused=t+1:size(eigvec, 2);
temp = im_mean(:)'*eigvec(:,unused); %row vec, b_j on Page 11
val = eigvec(:,unused)*temp';
reconstructed_training_methodtwo = bsxfun(@plus, reconstructed_training_methodtwo, val);
rec_error_training = 1/size(Xtrain,2)*sum(sum((Xtrain - reconstructed_training_methodtwo).^2,1)); %2.5773e+05 error is also the same using the second method!!!!

%Equivalent formuations of reconstruction error (Page 12 bottom, PCA
%lecture slides)
J = 0;
for idx = t+1:size(eigvec,2)
    J = J + eigvec(:,idx)'*R_hat*eigvec(:,idx);
end
disp(J) %2.5773e+05
1/size(Xtrain,2)*sum(sum(bsxfun(@minus,(Xtrain'*eigvec(:,unused)),im_mean(:)'*eigvec(:,unused)).^2)); %2.5773e+05

%% Classification: face recognition with nearest neighbour (NN) method 
close all;

M_PCA_vec = 33:10:363; %length is 34
neighbours_vec = 1:5;  
accuracy = zeros(34,5);
for i = 1:34
    for j = 1:5
        t = M_PCA_vec(i);
        neighbours = neighbours_vec(j);
        classknn = knnclassify(projection_testing(:,1:t), projection_training(:,1:t), ltrain', neighbours);
        res = sum((classknn==ltest')) / length(ltest);
        if res > max(accuracy(:))
            best_prediction = classknn;
        end
        accuracy(i,j) = res;
        %index = index + 1;
    end
end

% predictions = zeros(size(Xtest,2), max(ltrain));
% labels = zeros(size(predictions));
% for i = 1:size(Xtest,2)
%     predictions(i,classknn(i)) = 1;
%     labels(i,ltest(i)) = 1;
% end
%plotconfusion(labels', predictions'); %Not really visible

%Best option
[val, ix] = max(accuracy(:));
[best_M_PCA_idx, best_neighbours_idx] = ind2sub(size(accuracy), ix);
disp(val);
disp(M_PCA_vec(best_M_PCA_idx));
disp(neighbours_vec(best_neighbours_idx));

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
title(['Confusion matrix for PCA-NN classification using the best parameters, Accuracy = ', num2str(acc)], 'FontSize', 20)
xlabel('Predicted classes', 'FontSize', 16)
ylabel('Actual classes', 'FontSize', 16)
        

