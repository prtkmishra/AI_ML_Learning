%% Competetive Learning for clustering
% This script calculates clusters using competitive learning without
% normalization
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
X = [0.41
    0.76
    0.34
    0.06
    0.37]

C = [-0.5 0 1.5
    1.5 2.5 0]

eta = 0.1; % Learning Rate
order = [3 1 1 5 6]; % Order of selecting the samples for clustering

%% Start Clustering Algorithm
for i = 1:size(order,2)
    fprintf('#################################')
    fprintf(' For Iteration: %d ',i);
    fprintf('################################# \n')
    euc = [];
    for j = 1:size(C,2)
        euc = [euc norm((X(:,order(:,i))-C(:,j)))];
    end
    euc % Is the calculated euclidean distance from sample to all centroids
    for n = 1:size(euc,2)
        if euc(:,n) == min(euc)
            idx = n; % Is the argmin of euc dist.
        end
    end     
    Selected_Cluster = idx
    C(:,idx) = C(:,idx)+eta*(X(:,order(:,i))-C(:,idx));
    UpdatedCluster = C % Is the update cluster centers for this Iteration
end
