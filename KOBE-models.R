#install.packages("mlbench")
setwd('D:/5415/data.csv')
kb <- read.csv("data.csv", sep = "," ,stringsAsFactors = FALSE)
kb <- kb[!is.na(kb$shot_made_flag),]
head(kb)
#kb <- na.omit(kb)
#colSums(is.na(kb))
set.seed(123)
training <- sample(1:nrow(kb), nrow(kb) * .7)
str(training)

kb.train <- kb[training,]
kb.test <- kb[-training,]
kb.train$time_remaining <- kb.train$minutes_remaining*60+kb.train$seconds_remaining
kb.test$time_remaining <- kb.test$minutes_remaining*60+kb.test$seconds_remaining
kb.train$shot_distance[kb.train$shot_distance>40] <- 40
kb.test$shot_distance[kb.test$shot_distance>40] <- 40
for(i in 1:ncol(kb.train)){
  if(is.character(kb.train[, i])){
    kb.train[, i] <- as.factor(kb.train[, i])
    print(i)
  }
}
for(i in 1:ncol(kb.test)){
  if(is.character(kb.test[, i])){
    kb.test[, i] <- as.factor(kb.test[, i])
    print(i)
  }
}
for(i in 1:ncol(kb.test)){
  if(is.character(kb.test[, i])){
    kb.test[, i] <- as.factor(kb.test[, i])
    print(i)
  }
}
kb.train$shot_made_flag <- as.factor(kb.train$shot_made_flag)
kb.train$shot_made_flag <- factor(kb.train$shot_made_flag, levels = c("0", "1"))
str(kb.train)


library(mlbench)
library(caret)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(shot_made_flag~., data=kb.train, method="randomforest", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)


#RandomForest
library(randomForest)
kobeforest <- randomForest(as.factor(shot_made_flag) ~ combined_shot_type + period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining ,data=kb.train, ntree = 200, mtry = 5, nodesize = 1)
pred.kobe <- predict(kobeforest, kb.test,type="response")
table(kb.test$shot_made_flag, pred.kobe)
varImpPlot(kobeforest)
confusionMatrix(as.factor(kb.test$shot_made_flag), pred.kobe)

#glm
model <- glm(as.factor(shot_made_flag) ~ combined_shot_type  +period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining, family="binomial",data=kb.train)
pred.kobe.glm <- predict(model, kb.test,type="response")
mean(pred.kobe.glm)
prd<-as.numeric(pred.kobe.glm>0.5)
prd<-as.factor(prd)
table(kb.test$shot_made_flag, prd)
confusionMatrix(as.factor(kb.test$shot_made_flag), prd)

#svm
library(tidyverse)
library(gridExtra)
library(e1071)
wts=c(1,1)
names(wts)=c(1,0)
modelsvm <- svm(as.factor(shot_made_flag) ~ combined_shot_type  +period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining,data=kb.train, kernel="radial",  gamma=1, cost=1, class.weights=wts)
pred.kobe.svm <- predict(modelsvm, kb.test,type="response")
table(kb.test$shot_made_flag, pred.kobe.svm)
confusionMatrix(as.factor(kb.test$shot_made_flag), pred.kobe.svm)

#plot
plot(kobeforest)
library("party")
x <- ctree(as.factor(shot_made_flag) ~ combined_shot_type  +period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining,data=kb.train)
plot(x, type="simple")
tree <- getTree(kobeforest, k=1, labelVar=TRUE)
dev.off()
plot(modelsvm,kb.test,shot_distance~minutes_remaining)
plot(modelsvm,kb.test,loc_x~loc_y)
#roc
library(ggplot2)
rocobj <- roc(pred.kobe,kb.test$shot_made_flag)
rocobj2 <- roc(prd,kb.test$shot_made_flag)
rocobj3 <- roc(pred.kobe.svm,kb.test$shot_made_flag)
g2 <- ggroc(list(rocobj,rocobj2,rocobj3))
print(g2+ ggtitle("ROC")+labs(title="RandomForest","glm","svm"),print.auc = TRUE)




#KNN 
library(caret)
library(pROC)
library(mlbench)
library(kknn)

set.seed(200)
trControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats = 3)

set.seed(200)
fit.shot.20 <- train(as.factor(shot_made_flag) ~ 
                       combined_shot_type   +period + opponent + season+loc_x + loc_y  + shot_distance + minutes_remaining, 
                     data = kb.train,
                     method = 'knn',
                     tuneLength = 20,
                     trControl= trControl)
plot(fit.shot.20,main = 'fit.shot.20')
pred.knn.20 <- predict(fit.shot.20, kb.train)
confusionMatrix(pred.knn.20,as.factor(kb.train$shot_made_flag))

set.seed(200)
fit.shot.150 <- train(as.factor(shot_made_flag) ~ 
                        combined_shot_type +period + opponent + season + loc_x + loc_y  + shot_distance + minutes_remaining, 
                     data = kb.train,
                     method = 'knn',
                     tuneLength = 150,
                     trControl= trControl)
plot(fit.shot.150,main = 'fit.shot.150')
pred.knn.150 <- predict(fit.shot.150, kb.train)
confusionMatrix(pred.knn.150,as.factor(kb.train$shot_made_flag))

# way 2
kb.train$shot_made_flag <- as.factor(kb.train$shot_made_flag)
kb.test$shot_made_flag <- as.factor(kb.test$shot_made_flag)

model.knn <- kknn(shot_made_flag ~ 
                    combined_shot_type +period + opponent + season + loc_x + loc_y  
                  + shot_distance + minutes_remaining,
                  kb.train, kb.test, k=213)
fit.knn = fitted(model.knn)
confusionMatrix(fit.knn, kb.test$shot_made_flag)

# NN model
library(neuralnet)
library(nnet)

combined_shot_type.class <- class.ind(kb.train$combined_shot_type)
kb.train.nn <- data.frame(kb.train$shot_made_flag, combined_shot_type.class, kb.train$shot_distance/100,kb.train$minutes_remaining/11,kb.train$loc_x/100,kb.train$loc_y/200)
set.seed(300)
model.nn <- neuralnet(kb.train.shot_made_flag  ~ ., 
                      data = kb.train.nn, err.fct = "ce", 
                      linear.output = F, hidden = 3)
plot(model.nn)
pred.nn <- round(predict(model.nn, kb.train.nn[,-1]),0)
confusionMatrix(as.factor(pred.nn), as.factor(kb.train$shot_made_flag))

######################################################################################################
# evaluation 
str(kb.train)

for (i in 1:dim(kb.train)[2]){
  print(tapply(kb.train[,i], kb.train[,i], length))
  }

library(pROC) 
roc1<-roc(kb$minutes_remaining,kb$shot_distance, plot=TRUE, print.thres=TRUE, print.auc=TRUE)
plot(roc1,print.auc=TRUE,plot=TRUE, print.thres=TRUE)

######################################################################################################
library(glmnet)
library(ROCR)
kobeforest <- randomForest(as.factor(shot_made_flag) ~ combined_shot_type + loc_x + loc_y  + shot_distance + minutes_remaining,data=kb.train, ntree = 200, mtry = 5, nodesize = 1)
model <- glm(as.factor(shot_made_flag) ~ combined_shot_type + loc_x + loc_y  + shot_distance + minutes_remaining, family="binomial",data=kb.train)
modelsvm <- svm(as.factor(shot_made_flag) ~ combined_shot_type + loc_x + loc_y  + shot_distance + minutes_remaining, data=kb.train, kernel="radial",  gamma=1, cost=1, class.weights=wts)
model.knn <- kknn(shot_made_flag ~ combined_shot_type + loc_x + loc_y + shot_distance + minutes_remaining, kb.train, kb.test, k=213)
model.nn <- neuralnet(kb.train.shot_made_flag  ~ .,  data = kb.train.nn, err.fct = "ce", linear.output = F, hidden = 3)


fit1 <- glm(shot_made_flag ~ combined_shot_type + shot_zone_basic + season+ minutes_remaining, data = kb.train, family = binomial())
prob1 <- predict(fit1, newdata = kb.test, type ="response")
pred1 <- prediction(prob1,kb.test$shot_made_flag)
performance(pred1, 'auc')@y.values[[1]]
plot(performance(pred1,'tpr','fpr'),colorize = T, lwd = 3, main = 'ROC Curves')
abline(a=0, b=1, Ity =2, lwd = 3,col ='blue')



















