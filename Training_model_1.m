clc;clear
T = readtable('Maternal Health Risk Data Set.csv');
TF = isempty(T)%To check null


%% 
% split the data into training set and testing set

% Cross validation (train: 70%, test: 15%, validation: 15%)
% The idea is from 
% https://uk.mathworks.com/matlabcentral/answers/377839-split-training-data-and-testing-data
datasize = size(T,1);
trainSize = round(datasize * 0.7);
valSize = round(datasize * 0.15);
testSize = round(datasize *0.15);
cv = cvpartition(size(T,1),'HoldOut',0.15 + 0.15);
idx = cv.test;
% Separate to training and test+validation data
dataTrain = T(~idx,:);
dataValTest  = T(idx,:);
% seperate test and validation set
cv = cvpartition(valSize+testSize,'HoldOut',0.5);
idx = cv.test;
dataVal = T(~idx,:);
dataTest = T(idx,:);

dataTrain_X = dataTrain(:,1:6);
dataTrain_Y = dataTrain(:,"RiskLevel");

dataVal_X = dataVal(:,1:6);
dataVal_Y = dataVal(:,"RiskLevel");

dataTest_X = dataTest(:,1:6);
dataTest_Y = dataTest(:,"RiskLevel");

%%

% Generate an exponentially spaced set of values from 1 through 10 that 
% represent the minimum number of observations per leaf node.
leafs = logspace(0,1,10);% range from 10^0 to 10^1, 10 steps. (1-10)
% Create cross-validated classification trees. 
% Specify to grow each tree using a minimum leaf size in leafs.
rng('default')
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitctree(dataTrain_X,dataTrain_Y, ...
        'ClassNames',{'high risk','low risk','mid risk'},...
        'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');
% The idea of the above code is from
% Example in Improving Classification Trees and Regression Trees
% https://uk.mathworks.com/help/stats/improving-classification-trees-and-regression-trees.html#bsw6baj
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build the new model with optimized hyperparameter 
%from the result above we got the optimized hyperparameter
%Tuning the hyperparameter
tic
newMdl = fitctree(dataTrain_X,dataTrain_Y, ...
                  'ClassNames',{'high risk','low risk','mid risk'},...
                  'CrossVal','on', ...
                  'MinLeafSize',8)
toc                
loss1 = kfoldLoss(newMdl)
%% 
% Run this Section to use Training set
Truelabel = dataTrain_Y{:,:};
[predictedlabel,score] = predict(newMdl.Trained{10},dataTrain_X);
%% Make confusion matrix
%                   
cm = confusionchart(Truelabel,predictedlabel)
%% 
cmm = confusionmat(Truelabel,predictedlabel)
%% 

Total = sum(cmm,'all');
% Calculate TP, FP, TN, FN
% Confusion matrix details of high risk
TP_high = cmm(1,1);
FP_high = sum(cmm(:,1),1) - TP_high;
FN_high = sum(cmm(1,:),2) - TP_high;
TN_high = Total - TP_high - FP_high - FN_high;
% details of mid risk
TP_mid = cmm(3,3);
FP_mid = sum(cmm(:,3),1) - TP_mid;
FN_mid = sum(cmm(3,:),2) - TP_mid;
TN_mid = Total - TP_mid - FP_mid - FN_mid;
% details of low risk
TP_low = cmm(2,2);
FP_low = sum(cmm(:,2),1) - TP_low;
FN_low = sum(cmm(2,:),2) - TP_low;
TN_low = Total - TP_low - FP_low - FN_low;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here we caluclate the recall, precision and f1 score

Recall_high = TP_high/(TP_high + FN_high);
Recall_mid = TP_mid/(TP_mid + FN_mid);
Recall_low = TP_low/(TP_low + FN_low);


Recall = (Recall_high + Recall_mid + Recall_low)/3 %Sensitivity

%Precision
Pre_high = TP_high/(TP_high + FP_high);
Pre_mid = TP_mid/(TP_mid + FP_mid);
Pre_low = TP_low/(TP_low + FP_low);

Precision = (Pre_high + Pre_mid + Pre_low)/3

%Specificity
Spec_high = TN_high/(TN_high + FP_high);
Spec_mid = TN_mid/(TN_mid + FP_mid);
Spec_low = TN_low/(TN_low + FP_low);

Specificity = (Spec_high+Spec_mid+Spec_low)/3
%F1 Score
F1Score = (2 * (Precision * Recall)/(Precision + Recall))
%Accuracy
Acc_high = (TP_high+TN_high)/Total;
Acc_mid = (TP_mid+TN_mid)/Total;
Acc_low = (TP_low+TN_low)/Total;
Accuracy = (Acc_high+Acc_mid+ Acc_low)/3
%%
save("Final_model_1.mat",'newMdl')