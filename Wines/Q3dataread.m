%% Cw2 Part 3
clc, clear

%% Read data
data = dlmread('wine.data.csv');

train_data = data(data(:,1)==1,:);
test_data = data(data(:,1)==2,:);

x_train = train_data(:,3:end);
y_train = train_data(:,2);
x_test = test_data(:,3:end);
y_test = test_data(:,2);

t_train = zeros(length(y_train), 3);
idx = sub2ind(size(t_train), 1:length(y_train), y_train');
t_train(idx) = 1;

t_test = zeros(length(y_test), 3);
idx = sub2ind(size(t_test), 1:length(y_test), y_test');
t_test(idx) = 1;

X_merged = [x_train;x_test];
train_max_idx = size(x_train,1);
test_max_idx = size(x_test,1);
Y_merged = [t_train;t_test];

%% Cross-validation
layer_mat = cell(14,1);
layer_mat{1} = 10;
layer_mat{2} = 8;
layer_mat{3} = 6;
layer_mat{4} = 4;
layer_mat{5} = [10 8];
layer_mat{6} = [10 6];
layer_mat{7} = [10 4];
layer_mat{8} = [8 6];
layer_mat{9} = [8 4];
layer_mat{10} = [6 4];
layer_mat{11} = [10 8 6];
layer_mat{12} = [10 8 4];
layer_mat{13} = [10 6 4];
layer_mat{14} = [10 8 6 4];

%% Unnormalized, tanh transfer function

best_test_acc = 0;
test_acc = zeros(5,14);
train_acc = zeros(5,14);

for i = 1:5
    for j = 1:length(layer_mat)
        [test_acc(i,j), train_acc(i,j)] = Q3NNgeneratedScript(X_merged, Y_merged, train_max_idx, test_max_idx, y_train, y_test, layer_mat{j}, i, 1);
        if test_acc(i,j) > best_test_acc
            best_test_acc = test_acc(i,j);
            corresponding_train_acc = train_acc(i,j);
            best_optim_setting = i;
            best_layer_setting = j;
        end
    end
end

%% Normalized, tanh transfer function

X_meanzero = bsxfun(@minus, X_merged, mean(x_train, 1));
X_norm = bsxfun(@rdivide, X_meanzero, std(x_train, 1));


best_test_acc_2 = 0;
test_acc_2 = zeros(5,14);
train_acc_2 = zeros(5,14);

for i = 1:5
    for j = 1:length(layer_mat)
        [test_acc_2(i,j), train_acc_2(i,j)] = Q3NNgeneratedScript(X_norm, Y_merged, train_max_idx, test_max_idx, y_train, y_test, layer_mat{j}, i, 1);
        if test_acc_2(i,j) > best_test_acc_2
            best_test_acc_2 = test_acc_2(i,j);
            corresponding_train_acc_2 = train_acc_2(i,j);
            best_optim_setting_2 = i;
            best_layer_setting_2 = j;
        end
    end
end

%% Unnormalized, sigmoid transfer function

best_test_acc_3 = 0;
test_acc_3 = zeros(5,14);
train_acc_3 = zeros(5,14);

for i = 1:5
    for j = 1:length(layer_mat)
        [test_acc_3(i,j), train_acc_3(i,j)] = Q3NNgeneratedScript(X_merged, Y_merged, train_max_idx, test_max_idx, y_train, y_test, layer_mat{j}, i, 2);
        if test_acc_3(i,j) > best_test_acc_3
            best_test_acc_3 = test_acc_3(i,j);
            corresponding_train_acc_3 = train_acc_3(i,j);
            best_optim_setting_3 = i;
            best_layer_setting_3 = j;
        end
    end
end

%% Normalized, sigmoid transfer function

best_test_acc_4 = 0;
test_acc_4 = zeros(5,14);
train_acc_4 = zeros(5,14);

for i = 1:5
    for j = 1:length(layer_mat)
        [test_acc_4(i,j), train_acc_4(i,j)] = Q3NNgeneratedScript(X_norm, Y_merged, train_max_idx, test_max_idx, y_train, y_test, layer_mat{j}, i, 2);
        if test_acc_4(i,j) > best_test_acc_4
            best_test_acc_4 = test_acc_4(i,j);
            corresponding_train_acc_4 = train_acc_4(i,j);
            best_optim_setting_4 = i;
            best_layer_setting_4 = j;
        end
    end
end

%% Best result separately to produce graph

[test_acc_bestsetting, train_acc_bestsetting] = Q3NNgeneratedScript_view(X_merged, Y_merged, train_max_idx, test_max_idx, y_train, y_test, layer_mat{11}, 2, 1);

disp(test_acc_bestsetting)
disp(train_acc_bestsetting)

