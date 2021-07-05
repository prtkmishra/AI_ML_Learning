%% Princple Component Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is a basic implementation of PCA algorithm and 
% explains the step by step implementation of PCA using 
% Karhunen-Loève Transform method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the nxd matrix to be converted to n x k matrix
%%Input Variables
clc;
fprintf('#################################')
fprintf(' Input Parameters ');
fprintf('#################################')
% Input matrix given as below. where rows correspond to samples and
% columns correspond to dimensions. for e.g. (x1,x2) and (x3,x4) can be
% written as 
% X = [x1 x3
%     x2 x4]

% X = [1 2 3 2
%     2 3 5 2
%     1 1 1 1]

X = [2.8284   -2.8284    2.8284   -2.8284
    -1.4142   -1.4142    1.4142    1.4142]
% Define input Dimension n
n = size(X,1);

% Define output Dimension k
k = 1;

%% Calcuation
fprintf('#################################')
fprintf(' Calculate Covariance Matrix ');
fprintf('#################################')
% Compute the mean row matrix
mean_row_vector = mean(X,2)

% Subtract mean (hence, the zero-mean data is)
zeromeandata = X - mean_row_vector

% Compute the covariance matrix of rows of B (results in dxd matrix)
% C = B'*B;
CovarianceMatrix = (zeromeandata*zeromeandata')/size(X,2)

fprintf('#################################')
fprintf(' Identify eigen vectors and values ');
fprintf('#################################')
% Compute the k largest eigenvectors
[v,d] = eig(CovarianceMatrix)

fprintf('#################################')
fprintf(' Identify the k largest Eigen vectors ');
fprintf('#################################') 
% Compute matrix W of k-largest eigenvectors
w = [];
for i = 1:k
    w(:,i) = v(:,n+1-i);
end
w %Eigen vector

%% Output
fprintf('#################################')
fprintf(' Reduced Dimensions are ');
fprintf('#################################')
% Each column correspond to a sample with reduced dimension
Y = w'*zeromeandata
