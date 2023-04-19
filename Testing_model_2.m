
load("Final_model_2.mat")
%%
Truelabel = dataTest_Y{:,:};%convert testing set from table to array

% Use testing set on new model to get prediction
[predictedlabel,score] = predict(Mdl,dataTest_X);
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
figure
plot(rocObj)
