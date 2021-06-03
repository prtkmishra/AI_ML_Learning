%% KNN Classifier
% using euclidean distance
SAMPLES = [0.15 0.15 0.12 0.1 0.06
    0.35 0.28 0.2 0.32 0.25]

X = [0.1
    0.25]

for i = 1:size(SAMPLES,2)
    (X(1,1) - SAMPLES(1:1)).^2+(X(2,1) - SAMPLES(2:1)).^2
    euc = sqrt((X(1,1) - SAMPLES(1:1)).^2+(X(2,1) - SAMPLES(2:1)).^2)
end
