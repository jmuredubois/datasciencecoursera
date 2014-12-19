library(caret)
library(kernlab)
#library(RSNNS)

## the presence of #DIV/0! (yeah Excel) in the data caused many variables to be imported as strings 
setwd("/Volumes/mediaOSX/R/predmachlearn")
data <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!"), as.is=6:154)

data[1:5,]

barplot(table(data$classe))

inTrain = createDataPartition(data$classe, p = 6/10)[[1]]

training <- data[inTrain, 6:159]
testTrn <- data[-inTrain, 6:159]

#preProc <- preProcess(training, method="pca")

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
trainPC <- predict(preProc, trainSafeVar)

## train KNN model
model <- train(x=trainPC, y=data$classe[inTrain], method="knn", metric="Accuracy")

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
predTrain <- predict(model, trainPC)

## compare confusion matrices
confTrain <- confusionMatrix(predTrain, data$classe[inTrain])
confTestTrn <- confusionMatrix(predTestTrn, data$classe[-inTrain])


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