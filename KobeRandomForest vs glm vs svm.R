#install.packages("mlbench")
setwd('D:/5415/kobe_project')
kb <- read.csv("data.csv", sep = "," ,stringsAsFactors = FALSE)
kb <- kb[!is.na(kb$shot_made_flag),]
head(kb)
kb <- na.omit(kb)
colSums(is.na(kb))
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
kobeforest <- randomForest(as.factor(shot_made_flag) ~ combined_shot_type + period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining ,data=kb.train, ntree = 200, mtry = 8, nodesize = 1)
pred.kobe <- predict(kobeforest, kb.test,type="response")
table(kb.test$shot_made_flag, pred.kobe)
varImpPlot(kobeforest)
confusionMatrix(as.factor(pred.kobe),as.factor(kb.test$shot_made_flag))

#glm
model <- glm(as.factor(shot_made_flag) ~ combined_shot_type  +period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining, family="binomial",data=kb.train)
pred.kobe.glm <- predict(model, kb.test,type="response")
mean(pred.kobe.glm)
prd<-as.numeric(pred.kobe.glm>0.5)
prd<-as.factor(prd)
table(kb.test$shot_made_flag, prd)
confusionMatrix(as.factor(prd),as.factor(kb.test$shot_made_flag) )

#svm
library(tidyverse)
library(gridExtra)
library(e1071)
wts=c(1,1)
names(wts)=c(1,0)
modelsvm <- svm(as.factor(shot_made_flag) ~ combined_shot_type  +period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining,data=kb.train, kernel="radial",  gamma=1, cost=1, class.weights=wts)
pred.kobe.svm <- predict(modelsvm, kb.test,type="response")
table(kb.test$shot_made_flag, pred.kobe.svm)
confusionMatrix(as.factor(pred.kobe.svm),as.factor(kb.test$shot_made_flag) )

#plot
plot(kobeforest)
library("party")
x <- ctree(as.factor(shot_made_flag) ~ combined_shot_type  +period + opponent + season + loc_x + loc_y  + shot_distance + time_remaining,data=kb.train)
plot(x, type="simple")
tree <- getTree(kobeforest, k=1, labelVar=TRUE)
dev.off()
plot(modelsvm,kb.test,shot_distance~minutes_remaining)
plot(modelsvm,kb.test,loc_x~loc_y)



library('pROC')
library('ggplot2')
rocobj <- roc(pred.kobe,kb.test$shot_made_flag)
rocobj2 <- roc(prd,kb.test$shot_made_flag)
rocobj3 <- roc(pred.kobe.svm,kb.test$shot_made_flag)
g2 <- ggroc(list(rocobj,rocobj2,rocobj3))
print(g2+ ggtitle("ROC"),print.auc = TRUE)

