library(ggplot2)
library(plyr)
library(caret)
library(kernlab)
#library(RSNNS)

# move to path on my machine where the data is
setwd("/Volumes/mediaOSX/R/datascitoolbox_assign01/datasciencecoursera")

## the presence of #DIV/0! (yeah Excel) in the data caused many variables to be imported as strings 
data <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!"), as.is=6:154)

data[1:5,]

png("training_classes.png", width=640, height=480)
barplot(table(data$classe))
dev.off()
#ggsave("training_classes.png", width=640, height=480)

inTrain = createDataPartition(data$classe, p = 6/10)[[1]]

training <- data[inTrain, 6:159]
testTrn <- data[-inTrain, 6:159]

newW <- charmatch(training$new_window, c("yes","no"))

trainSafe <- training
trainSafe$new_window <- newW
# http://stackoverflow.com/a/3418192

## remove columns with more than 80% NA
naCols <- colSums(is.na(trainSafe))
goodCols <- as.vector(naCols)/dim(trainSafe)[2] < 0.2
nGoodCols <- sum(goodCols)
badColNamesNA <- names(trainSafe)
badColNamesNA <- badColNamesNA[!goodCols]
badColNamesNA <- badColNamesNA[!is.na(badColNamesNA)]

trainSafeNA <- trainSafe[, !names(trainSafe) %in% badColNamesNA ]

## remove zero variance columns SO Niubius and Juba http://stackoverflow.com/a/15069056/4371208
varCols <- apply(trainSafeNA, 2, var, na.rm=TRUE)
goodCols <- abs(as.vector(varCols)) > 1e-3
badColNamesVar <- names(trainSafeNA)
badColNamesVar <- badColNamesVar[!goodCols]
badColNamesVar <- badColNamesVar[!is.na(badColNamesVar)]

trainSafeVar <- trainSafeNA[, !names(trainSafeNA) %in% badColNamesVar ]
trainSafe <- NULL # free some memory
trainSafeNA <- NULL # free some memory

## perform PCA
preProc <- preProcess(trainSafeVar, thresh=0.85, method="pca")
message("PCA preprocessing simplified to ", preProc$numComp, " variables.")
trainPC <- predict(preProc, trainSafeVar)

## train KNN model
model <- train(x=trainPC, y=data$classe[inTrain], method="knn", metric="Accuracy")
# plot model
ggplot(model)
#ggsave("knn_model_ggplot.png", width=2.13, height=1,6)
message("Tuned k for knn model : ", model$bestTune)

# prediction on training set
predTrain <- predict(model, trainPC)
confTrain <- confusionMatrix(predTrain, data$classe[inTrain])
#message("Training confusion matrix : ", confTrain)
message("Training set accuracy : ", confTrain$overall[1])
message("Training set Kappa : ", confTrain$overall[2])

## cross-validation
cvFolds <- createFolds(1:dim(trainPC)[1],4)

i <- 0 
resFolds <- NULL
cvAccuracy <- NULL
cvKappa <- NULL
for(fold in cvFolds){
  trainFolds <- trainPC[-fold,]
  testFolds <- trainPC[fold,]
  
  ## train KNN model
  modelFolds <- train(x=trainFolds, y=data$classe[inTrain[-fold]], method="knn", metric="Accuracy")
  predFolds <- predict(modelFolds, testFolds)
  confFolds <- confusionMatrix(predFolds, data$classe[inTrain[fold]])
  save(confFolds, file="confFolds.rdt")
  i <- i + 1
  cvAccuracy[[i]] <- confFolds$overall[1]
  cvKappa[[i]] <- confFolds$overall[2]
  resFolds[[i]] <- confFolds
  save(resFolds, file="resFolds.rdt")
}
cvMeanAccuracy <- mean(cvAccuracy)
message("Cross-Validation average accuracy : ", cvMeanAccuracy)
cvAccuracyStd <- std(cvAccuracy)
message("Cross-Validation accuracy standard deviation: ", cvAccuracyStd)
cvMeanKappa <- mean(cvKappa)
message("Cross-Validation average Kappa : ", cvMeanKappa)
cvKappaStd <- std(cvKappa)
message("Cross-Validation Kappa standard deviation: ", cvKappaStd)


## predict values for test set
newW <- charmatch(testTrn$new_window, c("yes","no"))
testTrnSafe <- testTrn
testTrnSafe$new_window <- newW
testTrnNA <- testTrnSafe[, !names(testTrnSafe) %in% badColNamesNA ]
testTrnVar <- testTrnNA[, !names(testTrnNA) %in% badColNamesVar ]
testTrnSafe <- NULL # free some memory
testTrnNA <- NULL # free some memory

testTrnPC <- predict(preProc, testTrnVar)
predTestTrn <- predict(model, testTrnPC)

## compare confusion matrices
confTestTrn <- confusionMatrix(predTestTrn, data$classe[-inTrain])
message("Testing set accuracy : ", confTestTrn$overall[1])
message("Testing set Kappa : ", confTestTrn$overall[2])


# LOAD TESTING DATA
dttest <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!"), as.is=6:154)
testing <- dttest[, 6:159]
newW <- charmatch(testing$new_window, c("yes","no"))
testSafe <- testing
testSafe$new_window <- newW
testNA <- testSafe[, !names(testSafe) %in% badColNamesNA ]
testVar <- testNA[, !names(testNA) %in% badColNamesVar ]

testPC <- predict(preProc, testVar)
predTest <- predict(model, testPC)

answers <- as.vector(predTest)
source("pml_write_files.R")
pml_write_files(answers)