%% K-Means Clusering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script calculates the K-means clustering algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% Initialize cluster centres same as input samples format
numClusters = 2; % Number of clusters

C = [-1 5
    3 1]

itercount = 2; % Number of iterations to run for K-means clustering

%% Start the K-Means clustering
for iter = 1:itercount
    fprintf('#################################')
    fprintf(' Start of Iteration %d ',iter);
    fprintf('#################################')
    disp('***')
    clusterMap = []; % Initialize a cluster map
    for i = 1:size(X,2) % For every samples
        euc = []; % Initialize a euc matrix
        e = 0;
        for j = 1:size(C,2) % For every centroid
            euc = [euc norm((X(:,i)-C(:,j)))]; % Calculate EUC dist.
        end
        fprintf(' euclidean distance from sample %d',i);euc % Print the euc distance for the sample with each centroid
        for n = 2:size(euc,2)
            if euc(:,n)<euc(:,n-1)
                c = n;
            else
                c = n-1;
            end
        end
        clusterMap = [clusterMap c];
    end
    disp(' ****************************************');
    disp(' Cluster mapping for all the samples');
    disp(' ****************************************');
    X 
    clusterMap

    fprintf(' ********');
    fprintf(' New Centroids at Iter %d',iter)
    fprintf(' ********');
    
    % For samples belongong to same class, calculate the mean
    C = [];
    for cl = 1:numClusters
        Z = [];
        for l = 1:size(clusterMap,2)
            if clusterMap(:,l) == cl
                Z = [Z X(:,l)];
            end
        end
        C = [C mean(Z,2)]; % Append the new calculated centroids
    end
new_Centroids = C % Print the updated centroids    
fprintf('################################# END OF ITERATION ################################# \n')
end


            
    