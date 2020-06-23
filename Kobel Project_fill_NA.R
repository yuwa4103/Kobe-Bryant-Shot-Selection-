#install.packages("tidyverse")
#install.packages('gridExtra')
#install.packages("e1071")

library(tidyverse)
library(gridExtra)
library(knitr)
library(e1071)

setwd('D:/5415/data.csv')
df <- read.csv("data.csv", sep = "," ,stringsAsFactors = FALSE)
train <- df[!is.na(df$shot_made_flag),]
test <- df[is.na(df$shot_made_flag),]
train$shot_made_flag <- as.factor(train$shot_made_flag)
train$shot_made_flag <- factor(train$shot_made_flag, levels = c("1", "0"))

#comnibe the categories to shot_distance and time_remaining
train$shot_distance[train$shot_distance>40] <- 40
train$time_remaining <- train$minutes_remaining*60+train$seconds_remaining;

#normalize function
myNormalize <- function (target) {
  (target - min(target))/(max(target) - min(target))
}
train$shot_distance <- myNormalize(train$shot_distance)
train$time_remaining <- myNormalize(train$time_remaining)
kobe <- data.frame(train$shot_distance, train$time_remaining, train$shot_made_flag)
colnames(kobe) <- c("shot_distance", "time_remaining", "shot_made_flag")
head(kobe)

#handle with the test features
test$shot_distance[test$shot_distance>40] <- 40
test$time_remaining <- test$minutes_remaining*60+test$seconds_remaining;
test$shot_distance <- myNormalize(test$shot_distance)
test$time_remaining <- myNormalize(test$time_remaining)
test_kobe <- data.frame(test$shot_distance, test$time_remaining, test$shot_made_flag)
colnames(test_kobe) <- c("shot_distance", "time_remaining", "shot_made_flag")

#build svm model by train data
wts=c(1,1)
names(wts)=c(1,0)
model <- svm(shot_made_flag~shot_distance+time_remaining, data=kobe, kernel="radial",  gamma=1, cost=1, class.weights=wts)
#show accuracy by train dataset
pred1 <- predict(model, kobe,type="response")
table(kobe$shot_made_flag,pred1)
#glm
glmnew <- glm(shot_made_flag ~ ., family="binomial",data = kobe)
summary(glmnew)
kobe$myprediction <- predict(glmnew,kobe,type="response")
head(kobe$myprediction)
prd.th<-as.numeric(prd>0.5)
table(kobe$shot_made_flag,prd.th)


