## loading the necessary packages
require(cvTools); require(randomForest); require(caret); require(verification); require(tictoc)
require(e1071)

## loading the built features
load("~/features_wrist_16sec.rda")
load("~/features_hip_16sec.rda")
load("~/features_ankle_16sec.rda")
load("~/features_thigh_16sec.rda")
load("~/features_upper_arm_16sec.rda")

## cleaning data
features_ankle_16sec<-features_ankle_16sec[,-20]
features_wrist_16sec<-features_wrist_16sec[,-20]
features_thigh_16sec<-features_thigh_16sec[,-20]
features_hip_16sec<-features_hip_16sec[,-20]
features_upper_arm_16sec<-features_upper_arm_16sec[,-20]

ind1<-complete.cases(features_ankle_16sec)
ind2<-complete.cases(features_wrist_16sec)
ind3<-complete.cases(features_thigh_16sec)
ind4<-complete.cases(features_hip_16sec)
ind5<-complete.cases(features_upper_arm_16sec)

indd<-which(ind1 &ind2 & ind3 & ind4 &ind5)
features_ankle_16sec<-features_ankle_16sec[indd,]
features_wrist_16sec<-features_wrist_16sec[indd,]
features_thigh_16sec<-features_thigh_16sec[indd,]
features_hip_16sec<-features_hip_16sec[indd,]
features_upper_arm_16sec<-features_upper_arm_16sec[indd,]

data_new<-cbind(features_wrist_16sec)
un=unique(data_new$V1)
unn<-vector()
for(i in 1:93){
  temp=data_new[which(data_new$V1==un[i]),]
  unn[i]<-(length(unique(temp$V2)))
}
data_new<-data_new[-which(data_new$V1 %in% un[which(unn<4)]),]
colnames(data_new)[1:2]<-c("id","label")
for(i in 1:dim(data_new)[1]){
  if(data_new$label[i] %in% c("Lying on the bed","Lying on the floor")){
    data_new$label[i]="Lying"
  }
}

## random forest model for classification of individual activities
run_rf_model_acti<-function(train,validation,colname)
{
  unique(train$label)
  train<-train[,-which(colnames(train)=="id")]
  validation<-validation[,-which(colnames(validation)=="id")]
  colnams<-colnames(train)
  validation<-validation[,which(colnames(validation) %in% colnams)]
  unique(validation$label)
  ind=which(colnames(train)=="label")
  
  ## standardizing the features (to be similar to the features used in other models)
  mean_t<-vector();sd_t<-vector()
  for(k in 1:(dim(train)[2]-1)){
    mean_t[k]<-mean(train[,k], na.rm = TRUE)
    sd_t[k]<-sd(train[,k], na.rm = TRUE)
    train[,k]<-(train[,k]-mean_t[k])/sd_t[k]
  }
  for(k in 1:(dim(validation)[2]-1)){
    validation[,k]<-(validation[,k]-mean_t[k])/sd_t[k]
  }
  
  ## grid search for parameter tuning
  ntree_values <- c(50,100,200,500) ; mtry_values <- c(1,2,5) 
  result <- NULL
  result <- data.frame(matrix(ncol=3)) ; colnames(result) <- c("acc","numoftrees","mtry")  
  for(g in 1:length(ntree_values)){
    for(h in 1:length(mtry_values)){
      ntree = ntree_values[g] ; mtry = mtry_values[h]
      model = randomForest(x=train[,-ncol(train)],y=train[,ncol(train)],replace=TRUE,ntree=ntree,mtry=mtry) 
      outcome_pred=predict(model,validation[,-ind],type="response")
      a=confusionMatrix(outcome_pred,validation[,ind])
      mod_result <- data.frame(matrix(ncol=3)) ; colnames(mod_result) <- c("acc","numoftrees","mtry")
      mod_result$acc=a$overall[1]
      mod_result$numoftrees=ntree;
      mod_result$mtry=mtry;
      result=rbind(result,mod_result)
    }
  }
  result=na.omit(result);
  ntree=min(result[result$acc==max(result$acc,na.rm = TRUE),"numoftrees"])
  mtry=min(result[result$acc==max(result$acc,na.rm = TRUE),"mtry"])
  final_model=randomForest(x=train[,-ncol(train)],y=train[,ncol(train)],replace=TRUE,ntree=ntree,mtry=mtry,importance = TRUE)
  outcome_pred=predict(final_model,validation[,-ncol(validation)],type="response")
  mteric_final_model=calculate_report_acti(outcome_pred,validation[,ncol(validation)])
  return (list("model"=final_model,"acc"=result[result$acc==max(result$acc,na.rm = TRUE),"acc"],"metric"=mteric_final_model$metric, "ntree"=ntree, "mtry"=mtry))
}

## prediction function
predict_label_rf_acti<-function(data_new,colname)
{
  data_new<-cbind(data_new[,-which(colnames(data_new)=="label")], data_new[,which(colnames(data_new)=="label")])
  colnames(data_new)[ncol(data_new)]<-"label"
  un<-unique(data_new$id)
  proc_data<-data_new
  models_rf <- vector(mode = "list", length = 250)
  acc <- vector(mode = "list", length = 250)
  acc_test <- vector(mode = "list", length = 250)
  important_features <- vector(mode = "list", length = 250)
  complete_model<-vector(mode = "list", length = 250)
  complete_model_on_test_results<-data.frame(matrix(ncol=357));
  
  ## reporting the performance metrics seperately for all 32 activity classes
  colnames(complete_model_on_test_results)<-c("acc", "acc_l", "acc_h", "NIR", "p_val", "Sensitivity_1", "Specificity_1", "PPV_1", "NPV_1",
                                              "Precision_1", "Recall_1", "F1_1","Prevalence_1","DetectionRate_1", "DetectionPrevalence_1", "BalancedAccuracy_1","Sensitivity_2",
                                              "Specificity_2", "PPV_2", "NPV_2", "Precision_2", "Recall_2", "F1_2","Prevalence_2","DetectionRate_2", "DetectionPrevalence_2", 
                                              "BalancedAccuracy_2","Sensitivity_3", "Specificity_3", "PPV_3", "NPV_3", "Precision_3", "Recall_3", "F1_3","Prevalence_3",
                                              "DetectionRate_3", "DetectionPrevalence_3", "BalancedAccuracy_3","Sensitivity_4", "Specificity_4", "PPV_4", "NPV_4", "Precision_4", 
                                              "Recall_4", "F1_4","Prevalence_4","DetectionRate_4", "DetectionPrevalence_4", "BalancedAccuracy_4","Sensitivity_5", "Specificity_5", 
                                              "PPV_5", "NPV_5", "Precision_5", "Recall_5", "F1_5","Prevalence_5","DetectionRate_5", "DetectionPrevalence_5", "BalancedAccuracy_5",
                                              "Sensitivity_6", "Specificity_6", "PPV_6", "NPV_6", "Precision_6", "Recall_6", "F1_6","Prevalence_6","DetectionRate_6", 
                                              "DetectionPrevalence_6", "BalancedAccuracy_6","Sensitivity_7", "Specificity_7", "PPV_7", "NPV_7", "Precision_7", "Recall_7", "F1_7",
                                              "Prevalence_7","DetectionRate_7", "DetectionPrevalence_7", "BalancedAccuracy_7","Sensitivity_8", "Specificity_8", "PPV_8", "NPV_8", 
                                              "Precision_8", "Recall_8", "F1_8","Prevalence_8","DetectionRate_8", "DetectionPrevalence_8", "BalancedAccuracy_8","Sensitivity_9",
                                              "Specificity_9", "PPV_9", "NPV_9", "Precision_9", "Recall_9", "F1_9","Prevalence_9","DetectionRate_9", "DetectionPrevalence_9",
                                              "BalancedAccuracy_9","Sensitivity_10", "Specificity_10", "PPV_10", "NPV_10", "Precision_10", "Recall_10", "F1_10","Prevalence_10",
                                              "DetectionRate_10", "DetectionPrevalence_10", "BalancedAccuracy_10","Sensitivity_11", "Specificity_11", "PPV_11", "NPV_11",
                                              "Precision_11", "Recall_11", "F1_11","Prevalence_11","DetectionRate_11", "DetectionPrevalence_11", "BalancedAccuracy_11",
                                              "Sensitivity_12", "Specificity_12", "PPV_12", "NPV_12", "Precision_12", "Recall_12", "F1_12","Prevalence_12",
                                              "DetectionRate_12", "DetectionPrevalence_12", "BalancedAccuracy_12","Sensitivity_13", "Specificity_13", "PPV_13", "NPV_13",
                                              "Precision_13", "Recall_13", "F1_13","Prevalence_13","DetectionRate_13", "DetectionPrevalence_13", "BalancedAccuracy_13","Sensitivity_14",
                                              "Specificity_14", "PPV_14", "NPV_14", "Precision_14", "Recall_14", "F1_14","Prevalence_14",
                                              "DetectionRate_14", "DetectionPrevalence_14", "BalancedAccuracy_14","Sensitivity_15", "Specificity_15", "PPV_15", "NPV_15", 
                                              "Precision_15", "Recall_15", "F1_15","Prevalence_15","DetectionRate_15", "DetectionPrevalence_15", "BalancedAccuracy_15",
                                              "Sensitivity_16", "Specificity_16", "PPV_16", "NPV_16","Precision_16", "Recall_16", "F1_16","Prevalence_16","DetectionRate_16",
                                              "DetectionPrevalence_16", "BalancedAccuracy_16",
                                              "Sensitivity_17", "Specificity_17", "PPV_17", "NPV_17","Precision_17", "Recall_17", "F1_17","Prevalence_17","DetectionRate_17",
                                              "DetectionPrevalence_17", "BalancedAccuracy_17",
                                              "Sensitivity_18", "Specificity_18", "PPV_18", "NPV_18","Precision_18", "Recall_18", "F1_18","Prevalence_18","DetectionRate_18",
                                              "DetectionPrevalence_18", "BalancedAccuracy_18",
                                              "Sensitivity_19", "Specificity_19", "PPV_19", "NPV_19","Precision_19", "Recall_19", "F1_19","Prevalence_19","DetectionRate_19",
                                              "DetectionPrevalence_19", "BalancedAccuracy_19",
                                              "Sensitivity_20", "Specificity_20", "PPV_20", "NPV_20","Precision_20", "Recall_20", "F1_20","Prevalence_20","DetectionRate_20",
                                              "DetectionPrevalence_20", "BalancedAccuracy_20",
                                              "Sensitivity_21", "Specificity_21", "PPV_21", "NPV_21","Precision_21", "Recall_21", "F1_21","Prevalence_21","DetectionRate_21",
                                              "DetectionPrevalence_21", "BalancedAccuracy_21",
                                              "Sensitivity_22", "Specificity_22", "PPV_22", "NPV_22","Precision_22", "Recall_22", "F1_22","Prevalence_22","DetectionRate_22",
                                              "DetectionPrevalence_22", "BalancedAccuracy_22",
                                              "Sensitivity_23", "Specificity_23", "PPV_23", "NPV_23","Precision_23", "Recall_23", "F1_23","Prevalence_23","DetectionRate_23",
                                              "DetectionPrevalence_23", "BalancedAccuracy_23",
                                              "Sensitivity_24", "Specificity_24", "PPV_24", "NPV_24","Precision_24", "Recall_24", "F1_24","Prevalence_24","DetectionRate_24",
                                              "DetectionPrevalence_24", "BalancedAccuracy_24",
                                              "Sensitivity_25", "Specificity_25", "PPV_25", "NPV_25","Precision_25", "Recall_25", "F1_25","Prevalence_25","DetectionRate_25",
                                              "DetectionPrevalence_25", "BalancedAccuracy_25",
                                              "Sensitivity_26", "Specificity_26", "PPV_26", "NPV_26","Precision_26", "Recall_26", "F1_26","Prevalence_26","DetectionRate_26",
                                              "DetectionPrevalence_26", "BalancedAccuracy_26",
                                              "Sensitivity_27", "Specificity_27", "PPV_27", "NPV_27","Precision_27", "Recall_27", "F1_27","Prevalence_27","DetectionRate_27",
                                              "DetectionPrevalence_27", "BalancedAccuracy_27",
                                              "Sensitivity_28", "Specificity_28", "PPV_28", "NPV_28","Precision_28", "Recall_28", "F1_28","Prevalence_28","DetectionRate_28",
                                              "DetectionPrevalence_28", "BalancedAccuracy_28",
                                              "Sensitivity_29", "Specificity_29", "PPV_29", "NPV_29","Precision_29", "Recall_29", "F1_29","Prevalence_29","DetectionRate_29",
                                              "DetectionPrevalence_29", "BalancedAccuracy_29",
                                              "Sensitivity_30", "Specificity_30", "PPV_30", "NPV_30","Precision_30", "Recall_30", "F1_30","Prevalence_30","DetectionRate_30",
                                              "DetectionPrevalence_30", "BalancedAccuracy_30",
                                              "Sensitivity_31", "Specificity_31", "PPV_31", "NPV_31","Precision_31", "Recall_31", "F1_31","Prevalence_31","DetectionRate_31",
                                              "DetectionPrevalence_31", "BalancedAccuracy_31",
                                              "Sensitivity_32", "Specificity_32", "PPV_32", "NPV_32","Precision_32", "Recall_32", "F1_32","Prevalence_32","DetectionRate_32",
                                              "DetectionPrevalence_32", "BalancedAccuracy_32")
  results <- list()
  metric_allmodel <- data.frame(matrix(ncol=357)) ; colnames(metric_allmodel) <- colnames(complete_model_on_test_results)
  metric_model_this_run<-data.frame(matrix(ncol=357)); colnames(metric_model_this_run) <- colnames(complete_model_on_test_results)
  
  count <- 1;
  k <- 6
  folds_1<-cvFolds(length(unique(data_new$id)), K=k)
  actual<-vector(); predicted<-vector()
  for(j in 1:6){     
    test_data<-proc_data[which(proc_data$id %in% un[(folds_1$subsets[folds_1$which == j])]),] 
    develop_data<-proc_data[which(proc_data$id %in% un[(folds_1$subsets[folds_1$which != j])]), ]
    print("j=");print(j);
    new_data <- develop_data
    new_data <- new_data[sample(nrow(new_data)),]
    k <- 5 
    folds <- cvFolds(NROW(new_data), K=k)
    
    ## cross valdiation
    for(f in 1:k){
      print(f)
      train <- new_data[which(new_data$id %in% un[(folds$subsets[folds$which != f])]), ] #Set the training set
      validation <- new_data[which(new_data$id %in% un[(folds$subsets[folds$which == f])]), ] #Set the validation set
      newpred_rf<-run_rf_model_acti(train, validation, colname)
      models_rf[[count]]<-newpred_rf$model
      important_features[[count]]<-newpred_rf$model$importance;
      metric_allmodel=rbind(metric_allmodel,newpred_rf$metric)
      metric_model_this_run=rbind(metric_model_this_run, newpred_rf$metric)
      acc[[count]]=newpred_rf$acc[1]
      acc_test[[j]]=newpred_rf$acc[1]
      count=count+1;
    }
    
    temp_count<-which.max(unlist(acc_test))
    best_model=models_rf[temp_count]
    temp<-metric_model_this_run
    
    ind=which(colnames(test_data)=="label")
    actual_outcome_test<-(test_data[,ncol(test_data)])
    mean_d<-vector();sd_d<-vector()
    develop_data<-develop_data[,-which(colnames(develop_data)=="id")]
    test_data<-test_data[,-which(colnames(test_data)=="id")]
    
    ## standardizing the test data using the parameters from developing data
    for(k in 1:(dim(develop_data)[2]-1)){
      mean_d[k]<-mean(develop_data[,k], na.rm = TRUE)
      sd_d[k]<-sd(develop_data[,k],na.rm = TRUE)
      develop_data[,k]<-(develop_data[,k]-mean_d[k])/sd_d[k]
    }
    for(k in 1:(ncol(test_data)-1)){
      test_data[,k]<-(test_data[,k]-mean_d[k])/sd_d[k]
    }
    
    colnams<-colnames(develop_data)
    test_data<-test_data[,which(colnames(test_data) %in% colnams)]

    complete_model[[j]]<-randomForest(x=develop_data[,-ncol(develop_data)],y=develop_data[,ncol(develop_data)], replace=TRUE, ntree=newpred_rf$ntree, mtry=newpred_rf$mtry)
    complete_model_on_test<-predict(complete_model[[j]], as.data.frame(test_data[,-ncol(test_data)]), type = "response")
    temping<-calculate_report_acti(complete_model_on_test,actual_outcome_test)$metric
    complete_model_on_test_results<-rbind(complete_model_on_test_results,temping)
    actual<-c(actual, actual_outcome_test)
    predicted<-c(predicted, complete_model_on_test)
  }
  
  proc_data<-proc_data[,which(colnames(proc_data) %in% colnams)]
  best_model_rf=models_rf[which.max(unlist(acc))]
  imp_feature_rf=important_features[which.max(unlist(acc))]
  ind=which(colnames(proc_data)=="label")
  actual_outcome=proc_data[,ind]

  return(list("best_model"=best_model_rf,"metric_allmodel"=metric_allmodel,"imp_feature"=imp_feature_rf,
              "complete_model"=complete_model, "complete_model_on_test"=complete_model_on_test, "complete_model_on_test_results"=complete_model_on_test_results,
              "predicted"=predicted, "actual"=actual))
}

##  multi-class performance function
calculate_report_acti<-function(predicted, reported){
  d<-confusionMatrix(predicted, reported)
  acc<-d$overall[1]
  acc_l<-d$overall[3]
  acc_h<-d$overall[4]
  NIR<-d$overall[5]
  p_val<-d$overall[6]
  for(i in 1:32){
    assign(paste0("Sensitivity_",i),d$byClass[i,1])
    assign(paste0("Specificity_",i),d$byClass[i,2])
    assign(paste0("PPV_",i),d$byClass[i,3])
    assign(paste0("NPV_",i),d$byClass[i,4])
    assign(paste0("Precision_",i),d$byClass[i,5])
    assign(paste0("Recall_",i),d$byClass[i,6])
    assign(paste0("F1_",i),d$byClass[i,7])
    assign(paste0("Prevalence_",i),d$byClass[i,8])
    assign(paste0("DetectionRate_",i),d$byClass[i,9])
    assign(paste0("DetectionPrevalence_",i),d$byClass[i,10])
    assign(paste0("BalancedAccuracy_",i),d$byClass[i,11])
  }
  metric<-data.frame(acc, acc_l, acc_h, NIR, p_val, Sensitivity_1, Specificity_1, PPV_1, NPV_1,
                     Precision_1, Recall_1, F1_1,Prevalence_1,DetectionRate_1, DetectionPrevalence_1, BalancedAccuracy_1,Sensitivity_2,
                     Specificity_2, PPV_2, NPV_2, Precision_2, Recall_2, F1_2,Prevalence_2,DetectionRate_2, DetectionPrevalence_2, 
                     BalancedAccuracy_2,Sensitivity_3, Specificity_3, PPV_3, NPV_3, Precision_3, Recall_3, F1_3,Prevalence_3,
                     DetectionRate_3, DetectionPrevalence_3, BalancedAccuracy_3,Sensitivity_4, Specificity_4, PPV_4, NPV_4, Precision_4, 
                     Recall_4, F1_4,Prevalence_4,DetectionRate_4, DetectionPrevalence_4, BalancedAccuracy_4,Sensitivity_5, Specificity_5, 
                     PPV_5, NPV_5, Precision_5, Recall_5, F1_5,Prevalence_5,DetectionRate_5, DetectionPrevalence_5, BalancedAccuracy_5,
                     Sensitivity_6, Specificity_6, PPV_6, NPV_6, Precision_6, Recall_6, F1_6,Prevalence_6,DetectionRate_6, 
                     DetectionPrevalence_6, BalancedAccuracy_6,Sensitivity_7, Specificity_7, PPV_7, NPV_7, Precision_7, Recall_7, F1_7,
                     Prevalence_7,DetectionRate_7, DetectionPrevalence_7, BalancedAccuracy_7,Sensitivity_8, Specificity_8, PPV_8, NPV_8, 
                     Precision_8, Recall_8, F1_8,Prevalence_8,DetectionRate_8, DetectionPrevalence_8, BalancedAccuracy_8,Sensitivity_9,
                     Specificity_9, PPV_9, NPV_9, Precision_9, Recall_9, F1_9,Prevalence_9,DetectionRate_9, DetectionPrevalence_9,
                     BalancedAccuracy_9,Sensitivity_10, Specificity_10, PPV_10, NPV_10, Precision_10, Recall_10, F1_10,Prevalence_10,
                     DetectionRate_10, DetectionPrevalence_10, BalancedAccuracy_10,Sensitivity_11, Specificity_11, PPV_11, NPV_11,
                     Precision_11, Recall_11, F1_11,Prevalence_11,DetectionRate_11, DetectionPrevalence_11, BalancedAccuracy_11,
                     Sensitivity_12, Specificity_12, PPV_12, NPV_12, Precision_12, Recall_12, F1_12,Prevalence_12,
                     DetectionRate_12, DetectionPrevalence_12, BalancedAccuracy_12,Sensitivity_13, Specificity_13, PPV_13, NPV_13,
                     Precision_13, Recall_13, F1_13,Prevalence_13,DetectionRate_13, DetectionPrevalence_13, BalancedAccuracy_13,Sensitivity_14,
                     Specificity_14, PPV_14, NPV_14, Precision_14, Recall_14, F1_14,Prevalence_14,
                     DetectionRate_14, DetectionPrevalence_14, BalancedAccuracy_14,Sensitivity_15, Specificity_15, PPV_15, NPV_15, 
                     Precision_15, Recall_15, F1_15,Prevalence_15,DetectionRate_15, DetectionPrevalence_15, BalancedAccuracy_15,
                     Sensitivity_16, Specificity_16, PPV_16, NPV_16,Precision_16, Recall_16, F1_16,Prevalence_16,DetectionRate_16,
                     DetectionPrevalence_16, BalancedAccuracy_16,
                     Sensitivity_17, Specificity_17, PPV_17, NPV_17,Precision_17, Recall_17, F1_17,Prevalence_17,DetectionRate_17,
                     DetectionPrevalence_17, BalancedAccuracy_17,
                     Sensitivity_18, Specificity_18, PPV_18, NPV_18,Precision_18, Recall_18, F1_18,Prevalence_18,DetectionRate_18,
                     DetectionPrevalence_18, BalancedAccuracy_18,
                     Sensitivity_19, Specificity_19, PPV_19, NPV_19,Precision_19, Recall_19, F1_19,Prevalence_19,DetectionRate_19,
                     DetectionPrevalence_19, BalancedAccuracy_19,
                     Sensitivity_20, Specificity_20, PPV_20, NPV_20,Precision_20, Recall_20, F1_20,Prevalence_20,DetectionRate_20,
                     DetectionPrevalence_20, BalancedAccuracy_20,
                     Sensitivity_21, Specificity_21, PPV_21, NPV_21,Precision_21, Recall_21, F1_21,Prevalence_21,DetectionRate_21,
                     DetectionPrevalence_21, BalancedAccuracy_21,
                     Sensitivity_22, Specificity_22, PPV_22, NPV_22,Precision_22, Recall_22, F1_22,Prevalence_22,DetectionRate_22,
                     DetectionPrevalence_22, BalancedAccuracy_22,
                     Sensitivity_23, Specificity_23, PPV_23, NPV_23,Precision_23, Recall_23, F1_23,Prevalence_23,DetectionRate_23,
                     DetectionPrevalence_23, BalancedAccuracy_23,
                     Sensitivity_24, Specificity_24, PPV_24, NPV_24,Precision_24, Recall_24, F1_24,Prevalence_24,DetectionRate_24,
                     DetectionPrevalence_24, BalancedAccuracy_24,
                     Sensitivity_25, Specificity_25, PPV_25, NPV_25,Precision_25, Recall_25, F1_25,Prevalence_25,DetectionRate_25,
                     DetectionPrevalence_25, BalancedAccuracy_25,
                     Sensitivity_26, Specificity_26, PPV_26, NPV_26,Precision_26, Recall_26, F1_26,Prevalence_26,DetectionRate_26,
                     DetectionPrevalence_26, BalancedAccuracy_26,
                     Sensitivity_27, Specificity_27, PPV_27, NPV_27,Precision_27, Recall_27, F1_27,Prevalence_27,DetectionRate_27,
                     DetectionPrevalence_27, BalancedAccuracy_27,
                     Sensitivity_28, Specificity_28, PPV_28, NPV_28,Precision_28, Recall_28, F1_28,Prevalence_28,DetectionRate_28,
                     DetectionPrevalence_28, BalancedAccuracy_28,
                     Sensitivity_29, Specificity_29, PPV_29, NPV_29,Precision_29, Recall_29, F1_29,Prevalence_29,DetectionRate_29,
                     DetectionPrevalence_29, BalancedAccuracy_29,
                     Sensitivity_30, Specificity_30, PPV_30, NPV_30,Precision_30, Recall_30, F1_30,Prevalence_30,DetectionRate_30,
                     DetectionPrevalence_30, BalancedAccuracy_30,
                     Sensitivity_31, Specificity_31, PPV_31, NPV_31,Precision_31, Recall_31, F1_31,Prevalence_31,DetectionRate_31,
                     DetectionPrevalence_31, BalancedAccuracy_31,
                     Sensitivity_32, Specificity_32, PPV_32, NPV_32,Precision_32, Recall_32, F1_32,Prevalence_32,DetectionRate_32,
                     DetectionPrevalence_32, BalancedAccuracy_32)
  return (list("metric"=metric))
}







set.seed(1)
tic()
rf_results_16sec_acti_alldatafortest<-predict_label_rf_acti(data_new, label)
toc()

wrist_rf_16_recognition<-rf_results_16sec_acti_alldatafortest
save(file = "~/wrist_recognition_new_checking.rda", 
     wrist_rf_16_recognition)
