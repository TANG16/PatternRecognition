function [test_accuracy, train_accuracy] = Q3NNgeneratedScript_view(X_merged, Y_merged, train_max_idx, test_max_idx, y_train, y_test, hiddenLayerSize, optimFunc, activFunc)

setdemorandstream(391418381)

x = X_merged';
t = Y_merged';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
% help nntrain to see options

% Create a Pattern Recognition Network
%hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

if activFunc ~= 1
    for num = 1:length(net.layers)
        net.layers{num}.transferFcn = 'logsig'; %default is tansig
    end
end

if optimFunc == 1
    net.trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation.
elseif optimFunc == 2
    net.trainFcn = 'traincgp'; % Conjugate gradient backpropagation with Polak-Ribiere updates.
elseif optimFunc == 3
    net.trainFcn = 'trainbfg'; % BFGS quasi-Newton backpropagation.
elseif optimFunc == 4
    net.trainFcn = 'trainrp'; %RPROP backpropagation.
else 
    net.trainFcn = 'traingdx'; % Gradient descent w/momentum & adaptive lr backpropagation.
end

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
%net.divideFcn = 'dividerand';  % Divide data randomly
%net.divideMode = 'sample';  % Divide up every sample
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:train_max_idx;
net.divideParam.valInd = (train_max_idx+1):train_max_idx+test_max_idx;
% [trainInd,valInd,testInd] = ...
% divideind(size(X_merged, 1),1:train_max_idx,(train_max_idx+1):(train_max_idx+test_max_idx),[]);
%net.divideParam.testRatio = 0/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
%     'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
%e = gsubtract(t,y);
%performance = perform(net,t,y)
%tind = vec2ind(t);
%yind = vec2ind(y);
%percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
%trainTargets = t .* tr.trainMask{1};
%valTargets = t .* tr.valMask{1};
%testTargets = t .* tr.testMask{1};
%testTargets = t_test';
%predicted_test = net(x_test');
%trainPerformance = perform(net,trainTargets,y)
%valPerformance = perform(net,valTargets,y)
%testPerformance = perform(net,testTargets,y)
%testPerformance = perform(net,testTargets,predicted_test)

% Test accuracy
pred_test = y(:, (train_max_idx+1):end);
[~,pred_class_test] = max(pred_test, [], 1);
test_accuracy = sum(pred_class_test'==y_test)/length(y_test);

% Train accuracy
pred_train = y(:, 1:train_max_idx);
[~,pred_class_train] = max(pred_train, [], 1);
train_accuracy = sum(pred_class_train'==y_train)/length(y_train);


% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
