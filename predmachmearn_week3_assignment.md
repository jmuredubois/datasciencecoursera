Predicitive machine learning week 3 assignment
===================

# Explore data 
## Load caret and kernlab
`library(caret)`
`library(kernlab)`
## Load training data
`data <- read.csv("pml-testing.csv")`
19622 observations and 160 variables :
1. No way all 160 variables are needed -> PCA should help.
2. The amount of observations seems to justify a standard 60-40 data split for model training.
First, take a llook at what the variables are :
`data[1,]`
* X is just an index (not relevant for classification)
* user_name is not relevant for classification
* timestamps are also not relevant for classification
Looks that the first 5 columns of the dataset can safely be excluded from the training set.

Now, let's look at the classe values, to check if we can spot any bias:
`data$classe`
Whoa ! far too many values to look at in the console.
Plotting the distribution of classe values (with help from Roland on SO : http://stackoverflow.com/a/21639445)
`barplot(table(data$classe))`
![Training classes distribution](training_classes.png "Training classes distribution")
The training data may be slightly skewed towards class A, but this does not seem problematic since :
1. the skew is not that large
2. there is a fair chance that class "A" is the "at rest" class, which will probably also be the most frequent in the general case.