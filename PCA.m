%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is a basic implementation of PCA algorithm and 
% explains the step by step implementation of PCA using Karhunen-Loève Transform method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the nxd matrix to be converted to n x k matrix
% X = [1 2 1; 2 3 1; 3 5 1; 2 2 1];
X = [0 1; 3 5; 5 4; 5 6; 8 7; 9 7];
% figure(1), scatter(X(:,1),X(:,2),'o')
% Compute the mean row vector
x_mean_row_vector = [];
for i = 1:size(X,2)
    x_mean_row_vector(:,i) = sum(X(:,i))/length(X);
end
column_vecotr = ones(length(X));

% Compute the mean row matrix
x_mean_row_matrix = column_vecotr(:,1)*x_mean_row_vector;

% Subtract mean (obtain mean centred data)
B = X - x_mean_row_matrix

% Compute the covariance matrix of rows of B (results in dxd matrix)
C = B'*B;
% C = (B'*B)/size(X,1)
% Compute the k largest eigenvectors
[v,d] = eig(C)

% Enter the k largest Eigen vector manually
% Compute matrix W of k-largest eigenvectors
% w = [0.3375 0.7067; 0.9359 -0.3227 ; -0.1009  -0.6296];
% w = [0.4719 -0.8817; 0.8817 0.4719;0 0];
w = [-0.8309; -0.5564];

W = w';
Xtrans = B';
% Multiply each datapoint xi for i ∈ {1, 2, . . . , n} with W
Y = [];
for i = 1:length(B)
%     Y(i,:) = W*Xtrans(:,i);
    Y(i,:) = W*B(:,i);
end
Y
% figure(2), scatter(Y(:,1),Y(:,2))
