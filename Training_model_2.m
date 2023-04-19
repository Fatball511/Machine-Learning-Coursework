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
% Using oobError function of TreeBagger model to plot the 
% OOB Error over the number of grown tree

Mdl1 = TreeBagger(100,dataTrain,"RiskLevel",OOBPrediction="on")

plot(oobError(Mdl1))

xlabel("Number of Grown Tree")
ylabel("Out-of-bag Classification Error")
%%
% Base on the result of the plot, tune number of trees to 95
tic
Mdl = fitcensemble(dataTrain_X,dataTrain_Y, ...
    'ClassNames',{'high risk','low risk','mid risk'}, ...
     'Method','Bag',...
       'NumLearningCycles',95)
toc
%% Run section to choose dataset (training/tesing/validation)
% Run this Section to use Training set
Truelabel = dataTrain_Y{:,:};
[predictedlabel,score] = predict(Mdl,dataTrain_X);  
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
%% ROC

rocObj = rocmetrics(Truelabel,score,Mdl.ClassNames);
AVG_AUC = sum(rocObj.AUC)/3
%%
save("Final_model_2.mat",'Mdl')