%% Fuzzy K-means Clustering

clc;
%% Input Parameters
fprintf('#################################')
fprintf(' Input Parameters ');
fprintf('#################################')
% Input matrix given as below. where rows correspond to samples and
% columns correspond to dimensions. for e.g. (x1,x2) and (x3,x4) can be
% written as 
% X = [x1 x3
%     x2 x4]
X = [-1 1 0 4 3 5
    3 4 5 -1 0 1]

% Configure the initial membership values. format (r,c) = (class,sample)
mu = [1 0.5 0.5 0.5 0.5 0
    0 0.5 0.5 0.5 0.5 1]

k = 2; % Number of clusters
b = 2; % 

Thresh = 0.5; % Threshold for convergence
conv = 0; % Initialize convergence check as 0
%% Normalize the membership
norm_mu = mu./sum(mu,1);

%% Run clustering
New_Centroids = 0;
while conv == 0
    fprintf('#################################')
    fprintf(' Iteration %d ',iter);
    fprintf('################################# \n')
    sq_nomr_mu = norm_mu.^2;
    sumsquared = sum(sq_nomr_mu,2);
    %% Calculate the cluster centres
    fprintf('#################################')
    fprintf(' New Centroids ');
    fprintf('################################# \n')
    M = [];
    for i = 1:k
        N = 0;
        for j = 1:size(X,2)
            N = N+ norm_mu(i,j)^b*X(:,j);
        end
        M = [M N./sumsquared(k,:)];
    end
    % Check difference
    Diff = M - New_Centroids
    New_Centroids = M
    if max(Diff)<Thresh
        fprintf('#################################')
        fprintf(' Algorithm has converged ');
        fprintf('################################# \n')
        conv = 1;
        break
    end
    %% Update the membership values
    fprintf('#################################')
    fprintf(' Updated membership Values ');
    fprintf('################################# \n')
    for j = 1:size(norm_mu,1)
        for i = 1:size(X,2)
            norm_mu(j,i) = ((1/norm(X(:,i)-M(:,j)))^2/((1/norm(X(:,i)-M(:,1)))^2+(1/norm(X(:,i)-M(:,2)))^2));
        end
    end
    Updated_membershipValues = norm_mu
end