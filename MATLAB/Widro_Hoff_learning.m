%% Widrow-Hoff Learning
% This script can be used for Perceptron Learning Algorithm for binary
% classification
clc;
%% System Variables
MODE = 1;            %0: Batch, 1 - Sequential
SAMPLE_NORM = 1;     % 0- Off, 1 - ON

%% Input Variables
% Input sample is the column in X. For e.g., (1,5) (2,5) (4,1) (5,1) will
% be written as below in matrix form
% X = [1 2 4 5
%     5 5 1 1]
% And class of the samples be defined as below
% class = [1 1 -1 -1]
% Weights initialised as below column vector
% a = [-25
%     6 
%     3]

fprintf('#################################')
fprintf(' Input Variables ');
fprintf('#################################')

X = [0 1 2 0 1
    0 0 1 1 2]

class = [1 1 1 -1 -1]
a = [-1.5
    5
    -1]

b = [2
    2
    2
    2
    2]
epoch_count = 2; % Number of Iterations that need to be run
learningRate = 0.2; % Learning Rate eta
%% Calculate the augmented notation vectors
fprintf('#################################')
fprintf(' Augment Vectors ');
fprintf('#################################')
% convert to augmented notation
Y = []; % Inititalize empty matrix
for i =1:size(class,2)
    if class(:,i) ~= 1 % Check for class label
        if SAMPLE_NORM == 1
            C = [1; X(:,i)]; % Append -ve of updated X in Y
            Y = [Y -1*C];% Append Updated X in Y
        else
            C = [1; X(:,i)]; % Append -ve of updated X in Y
            Y = [Y 1*C];% Append Updated X in Y
        end
    else
        C = [1;X(:,i)];% Append 1 in X
        Y = [Y C];% Append Updated X in Y
    end
end
% Print Final Y
Y

%% Start Training for every epoch for SAMPLE NORMALIZED DATA
if SAMPLE_NORM == 1
    for epoch = 1:epoch_count
        fprintf('#################################')
        fprintf(' This is epoch %d ',epoch);
        fprintf('#################################')
        if MODE == 0 % If mode is Batch learning      
            gx = a'*Y; % cost function
            l = 0;
            for i = 1:size(X,2) % Loop for every sample
                l = l+learningRate*(b(i,:)-(a'*Y(:,i)))*Y(:,i); % Update the weight vector
            end
            a = a+l;
%             Weights_of_current_Epoch = a %     Print updated weights for current epoch
        else % If mode is Sequential Learning
            for i = 1:size(X,2)% Loop for every sample
                a = a+learningRate*(b(i,:)-(a'*Y(:,i)))*Y(:,i) % Update the weight vector
            end
        end
        Weights_of_current_Epoch = a %     Print updated weights for current epoch
    end 

%% Start Training for every epoch for without SAMPLE NORMALIZATION
else
    for epoch = 1:epoch_count
        if MODE == 0 % If mode is Batch learning      
            gx = a'*Y; % cost function
            l = 0;
            for i = 1:size(X,2) % Loop for every sample
                if gx(:,i) <= 0 % If cost of the sample is less than or equal to zero
                    l = l+class(:,i)*learningRate*Y(:,i); % Update the weight vector
                end
                a = a+l;
            end
            Weights_of_current_Epoch = a %     Print updated weights for current epoch
        else % If mode is Sequential Learning
            for i = 1:size(X,2)% Loop for every sample
                gx = a'*Y(:,i); % cost function
                if sign(gx) ~= class(:,i) % If cost of the sample is less than or equal to zero
                    a = a+class(:,i)*learningRate*Y(:,i); % Update the weight vector
                end
            end
        end
    end
end
%% Final Weights are
fprintf('#################################')
fprintf(' Final Weights are: ');
fprintf('#################################')
Final_Weights = a


