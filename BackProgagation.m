%% Backpropagation in a FeedForward Neural Network
% This script calculates updated weights in a MLP using back propagation.
% This is a 2-3-1 NN
% Comment the section in update rules where update is not needed
%% Input Parameters
clc;
% The input is given as a column vector
X = [0.1
    0.9]

% The weight matrix is given in matrix form. Where row belongs to weights
% from input nodes to a single hidden node.
% Wji = [w11 w12
%     w21 w22]
Wji = [0.5 0 
    0.3 -0.7]

% Weights for bias from input to hidden node
% wjzero = [w10
%     w20]
Wjzero = [0.2
    0]

% Weights from hidden nodes to output nodes
% wkj = [m11 m12]
Wkj = [0.8 1.6]

% Weights from hidden bias node to output nodes
% Wkzero = [m10]
Wkzero = [-0.4]

t = 0.5;
n = 0.25;

ACTLayerHidden = 0; % 0-tansig, 1-logsig, 2-radb, 3-linear
ACTLayerOutput = 0; % 0-tansig, 1-logsig, 2-radb, 3-linear

iter = 1; % Number of iterations

%% Calculate Feed Forward
for i = 1:iter
    fprintf('#################################')
    fprintf(' Iteration %d ',i);
    fprintf('################################# \n')
%     First Hidden Layer
    fprintf('#################################')
    fprintf(' Hidden Layer output ');
    fprintf('#################################')
    netj = Wji*X + Wjzero
    if ACTLayerHidden == 0
        Y = tansig(netj) % Output of hidden layer 1
    end
    if ACTLayerHidden == 1
        Y = logsig(netj) % Output of hidden layer 1
    end
    if ACTLayerHidden == 2
        Y = 'TBD' % Output of hidden layer 1
    end
    if ACTLayerHidden == 3
        Y = netj % Output of hidden layer 1
    end
    
%     Output Layer
    fprintf('#################################')
    fprintf(' Final output ');
    fprintf('#################################')
    netk = Wkj*Y+Wkzero
    if ACTLayerOutput == 0
        Z = tansig(Wkj*Y+Wkzero) % Calculated output
    end
    if ACTLayerOutput == 1
        Z = logsig(Wkj*Y+Wkzero) % Calculated output
    end
    if ACTLayerOutput == 2
        Z = 'TBD' % Calculated output
    end
    if ACTLayerOutput == 3
        Z = Wkj*Y+Wkzero % Calculated output
    end
    error = t-Z
%% Run Backpropagation
%     Run backpropagation update on all weights
    fprintf('#################################')
    fprintf(' BackPropagation Output ');
    fprintf('#################################')    
    % Output Layer
    mkj = Wkj;
    for i = 1:size(Wkj,1)
        for j = 1:size(Wkj,2)
            % Select activation function
            if ACTLayerHidden == 0
                deltam = (4*exp(-2*netk)/((1+exp(-2*netk))^2));
            end
            if ACTLayerHidden == 1
                deltam = (exp(-1*netk)/((1+exp(-1*netk))^2));
            end
            if ACTLayerHidden == 2
                deltam = -2*exp(-1*netk^2)*netk;
            end
            if ACTLayerHidden == 3
                deltam = 1;
            end
            mkj(i,j) = mkj(i,j)+(-n*(Z-t)*deltam*mkj(:,i));
        end
    end
    % Bias Weight from hidden layer to output node(s)
    for i = 1:size(Wkzero,1)
        % Select activation function
            if ACTLayerHidden == 0
                deltam = (4*exp(-2*netk)/((1+exp(-2*netk))^2));
            end
            if ACTLayerHidden == 1
                deltam = (exp(-1*netk)/((1+exp(-1*netk))^2));
            end
            if ACTLayerHidden == 2
                deltam = -2*exp(-1*netk^2)*netk;
            end
            if ACTLayerHidden == 3
                deltam = 1;
            end
        Wkzero(i,:) = Wkzero(i,:)+(-n*((Z-t)*deltam));
    end
    
    % First Hidden Layer
    tempWji = Wji;
    for i = 1:size(Wji,1)
        for j = 1:size(Wji,2)
            if ACTLayerHidden == 0
                deltaw = (4*exp(-2*netj(i,:))/((1+exp(-2*netj(i,:)))^2));
            end
            if ACTLayerHidden == 1
                deltaw = (exp(-1*netj(i,:))/((1+exp(-1*netj(i,:)))^2));
            end
            if ACTLayerHidden == 2
                deltam = -2*exp(-1*netj(i,:)^2)*netj(i,:);
            end
            if ACTLayerHidden == 3
                deltam = 1;
            end
            tempWji(i,j) = tempWji(i,j)+(-n*(Z-t)*deltam*Wkj(:,i)*deltaw*X(j,:));
        end
    end
    
    % Bias weight from input to hidden layer
    for i = 1:size(Wjzero,1)
            if ACTLayerHidden == 0
                deltaw = (4*exp(-2*netj(i,:))/((1+exp(-2*netj(i,:)))^2));
            end
            if ACTLayerHidden == 1
                deltaw = (exp(-1*netj(i,:))/((1+exp(-1*netj(i,:)))^2));
            end
            if ACTLayerHidden == 2
                deltaw = -2*exp(-1*netj(i,:)^2)*netj(i,:);
            end
            if ACTLayerHidden == 3
                deltaw = 1;
            end
            Wjzero(i,:) = Wjzero(i,:)+(-n*(Z-t)*deltam*Wkj(:,i)*deltaw);
    end
    Wkzero % Updated bias
    Wkj = mkj % Updated hidden to output weights
    Wjzero % Updated bias
    Wji = tempWji % Updated input to hidden node
end