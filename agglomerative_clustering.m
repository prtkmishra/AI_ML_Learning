%% Agglomerative clustering
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
X = [-1 1 0 4 5 4
    3 2 1 0 4 2]

%% Calculate Proximity Matrix
prox = [];
for i = 1:size(X,2)
    for j = 1:size(X,2)
        prox(i,j) = norm((X(:,i)-X(:,j)).^2);
    end
end
prox

%% Calculate linkage
%     Z = linkage(X,METHOD) creates a hierarchical cluster tree using the
%     the specified algorithm. The available methods are:
%  
%        'single'    --- nearest distance (default)
%        'complete'  --- furthest distance
%        'average'   --- unweighted average distance (UPGMA) (also known as
%                        group average)
%        'weighted'  --- weighted average distance (WPGMA)
%        'centroid'  --- unweighted center of mass distance (UPGMC)
%        'median'    --- weighted center of mass distance (WPGMC)
%        'ward'      --- inner squared distance (min variance algorithm)
%  
Y = squareform(prox);
Z = linkage(Y,'average')
dendrogram(Z)