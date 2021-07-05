%% Oja's Learning Rule
% Oja's learning rule can be used for PCA where Y is the projected data.
% In this script we have 2-D input samples projected to 1-D output data.

%% Input parameters
fprintf('#################################')
fprintf(' Input variables ');
fprintf('#################################')

% Input data is given as below:for 2 samples (x1,x2) and (x3,x4) the input
% is
% X = [ x1 x3
%     x2 x4]
X = [0 3 5 5 8 9
    1 5 4 6 7 7]
% Initialise weights in a row vector
W = [-1 0]

epoch = 2; % Number of iterations for the learning algorithm
eta = 0.01; % Learning rate

%% Calculate zero-mean matrix
fprintf('#################################')
fprintf(' zero-mean ');
fprintf('#################################')
meanX = mean(X,2)
zeroX = X-meanX

%% Train the algorithm
for e = 1:epoch
    Y = [];
    for i = 1:size(X,2)
        Y = [Y W*zeroX(:,i)];
    end
    deltaW = 0;
    for j = 1:size(Y,2)
        deltaW = deltaW+eta*Y(:,j)*(zeroX(:,j)'-Y(:,j)*W);
    end
    fprintf('#################################')
    fprintf(' Weights at the end of Epoch %d ',e);
    fprintf('#################################')
    W = W+deltaW
end


