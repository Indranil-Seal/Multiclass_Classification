rm(list = ls())
options(scipen = 999)
setwd("C:\\Users\\AXIOM\\Desktop\\data\\Wine Quality Data")

library(nnet)
library(dplyr)
library(caret)


# Functions:
GetMode = function(D){
  UniqD = unique(D)
  UniqD[which.max(tabulate(match(D,UniqD)))]
}

# .csv --> data.frame
data = read.csv("wine_train.csv",header = T,stringsAsFactors = F)

# data_structure:
structure = data.frame()
for(i in 1:ncol(data)){
  colnames(data)[i] = gsub("\\.","_",colnames(data)[i])
  datum = data.frame(col = colnames(data)[i],
                     type = typeof(data[,i]),
                     uniq_ = length(unique(data[,i])),
                     uniq_values = ifelse(length(unique(data[,i])) < 6,paste(unique(data[,i]),collapse = "-"),"Too Many Entries"),
                     na_ = sum(is.na(data[,i])),
                     null_ = sum(is.null(data[,i])),
                     nan_ = sum(is.nan(data[,i])),
                     mean_ = ifelse(typeof(data[,i]) != "character",mean(data[,i],na.rm = T),"Not Applicable"),
                     median_ = ifelse(typeof(data[,i]) != "character",median(data[,i],na.rm = T),"Not Applicable"),
                     mode_ = GetMode(data[,i]),
                     Q005 = quantile(data[,i],0.05,na.rm = T),
                     Q095 = quantile(data[,i],0.95,na.rm = T))
  structure = rbind.data.frame(structure,datum,make.row.names = F)
  print(paste0("Column :",colnames(data)[i]))
  rm(datum)
}

# EDA

## Testing for Outliers : 
outliers = data.frame()
for(i in 1:(ncol(data)-1)){
  D1 = abs((min(data[,i],na.rm = T) - median(data[,i],na.rm = T)))+1
  D2 = abs((median(data[,i],na.rm = T) - max(data[,i],na.rm = T)))
  
  datum = data.frame(col = colnames(data)[i],
                     dev1 = D1,
                     dev2 = D2,
                     outlier_indicator = ifelse(D1-D2 >=5 | D1-D2 < -5,"Presence Indicated","None"))
  outliers = rbind.data.frame(outliers,datum,make.row.names = F)
  print(paste0("Examining :",colnames(data)[i]))
  rm(datum)
}

# Capping with Q95
cols = paste0(outliers[outliers$outlier_indicator == "Presence Indicated",]$col)
for(i in 1:length(cols)){
  data[,cols[i]] = ifelse(data[,cols[i]] >= quantile(data[,cols[i]],.95,na.rm = T),quantile(data[,cols[i]],.95,na.rm = T),data[,cols[i]])
}



# Central Tendency :
exploration = data.frame()
index = unique(data$quality[order(data$quality,decreasing = F)])

for(i in 1:(ncol(data)-1)){
  datum = data.frame(col = colnames(data[,c(1:11)])[i],
                     min_Q3 = min(data[data$quality == "3",i]),
                     mean_Q3 = mean(data[data$quality == "3",i]),
                     max_Q3 = max(data[data$quality == "3",i]),
                     min_Q4 = min(data[data$quality == "4",i]),
                     mean_Q4 = mean(data[data$quality == "4",i]),
                     max_Q4 = max(data[data$quality == "4",i]),
                     min_Q5 = min(data[data$quality == "5",i]),
                     mean_Q5 = mean(data[data$quality == "5",i]),
                     max_Q5 = max(data[data$quality == "5",i]),
                     min_Q6 = min(data[data$quality == "6",i]),
                     mean_Q6 = mean(data[data$quality == "6",i]),
                     max_Q6 = max(data[data$quality == "6",i]),
                     min_Q7 = min(data[data$quality == "7",i]),
                     mean_Q7 = mean(data[data$quality == "7",i]),
                     max_Q7 = max(data[data$quality == "7",i]),
                     min_Q8 = min(data[data$quality == "8",i]),
                     mean_Q8 = mean(data[data$quality == "8",i]),
                     max_Q8 = max(data[data$quality == "8",i]),
                     min_Q9 = min(data[data$quality == "9",i]),
                     mean_Q9 = mean(data[data$quality == "9",i]),
                     max_Q9 = max(data[data$quality == "9",i]))
  exploration = rbind.data.frame(exploration,datum,make.row.names = F)
  print(paste0("exploring :",colnames(data)[i]))
  rm(datum)
}

#
data$quality = as.factor(data$quality)

# Test Train Split :
require(dplyr)
data = data %>% mutate(id = row_number())
train = data %>% sample_frac(.70)
test = data %>% anti_join(train,by = 'id')
data$id = NULL; train$id = NULL; test$id = NULL


# Multinomal Logistic Regression : Max Iter = 1000 --> Converged at Iter 160, Value = 2741.261100
set.seed(123)
mlogit = multinom(quality ~., data = train,maxit = 1000)
model_summary = summary(mlogit)

# Calculating the Z Scores & p-values :
z = model_summary$coefficients/model_summary$standard.errors
p = (1-pnorm(abs(z),0,1))*2

print(z,digits = 2)
print(p,digits = 2)

# Model Study & Variable Selection Methods :
features = data.frame()
index = unique(data$quality[order(data$quality,decreasing = F)])
for(i in 1:nrow(z)){
  datum = as.data.frame(rbind(model_summary$coefficients[i,],
                              model_summary$standard.errors[i,],
                              z[i,],
                              p[i,]))
  datum$quality = rep(paste0(index[i+1],"-given",index[1]),nrow(datum))
  datum$attributes = c("Coefficients","Standard Errors","Z Score","p-value")
  datum = datum[,c("quality","attributes",colnames(datum)[2:12])]
  
  features = rbind.data.frame(features,datum,make.row.names = F)
  print(paste0("Feature Properties for quality :",index[i+1]))
  rm(datum)
}

# Selecting Features based on p-values
significant_features = data.frame()
for(i in 1:length(unique(features$quality))){
  dat = features[features$quality == unique(features$quality)[i] & features$attributes == "p-value",]
  f = colnames(dat)[dat[1,] < .05]
  datum = data.frame(quality = dat[1,1],
                     significant_features = f)
  significant_features = rbind.data.frame(significant_features,datum,make.row.names = F)
  print(paste0("Feature Selection for :",dat[1,1]))
  rm(dat,datum,f)
}

# Odds Ratio : Studying the Relative Risks 
oddsML = exp(coef(mlogit))
print(oddsML,digits = 2)

# Testing the Model Developed :
prediction = as.data.frame(predict(mlogit,test,type = "prob"))
prediction$predicted_class = colnames(prediction)[apply(prediction,1,which.max)]
test = cbind.data.frame(test,prediction$predicted_class)

#Confusion Matrix & Relative Importance of Variables:
require(caret)
confusionMatrix(as.factor(test[,13]),as.factor(test[,12]))

IV = varImp(mlogit)
IV$Variables = rownames(IV)
IV = IV[order(IV$Overall,decreasing = T),]
vars = unique(IV$Variables)[1:4] # Shortlisted Variables (Top 4)

########################## Lean MLR- Using the Shortlisted Variables ######################
mlogit_lean = multinom(quality ~ density+ chlorides + volatile_acidity + pH,data = train,maxit = 1000)
