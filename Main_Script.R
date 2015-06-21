library(caret)
library(kernlab)
library(readr)
library(plyr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
    
    
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url,destfile="pml-training.csv")
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url,destfile="pml-testing.csv")
raw_training <- read.csv("pml-training.csv",stringsAsFactors=FALSE)
raw_testing <- read.csv("pml-testing.csv",stringsAsFactors=FALSE)


##### Part 1. Clean data     #########
# Remove irrelevant columns - names, identifiers, date/time stamps, etc.

# Function to count number of NA values in a column
na_count <- function(x) {
    sum(is.na(x))
}

# Identify columns to remove
# 1. Identifiers and time stamps
rm_cols <- 1:7

# 2. Columns with missing values
missing_values <- apply(raw_training,2,na_count)

var_names <- names(missing_values)
var_valid <- data.frame(var_names=var_names,missing_count=missing_values,stringsAsFactors=FALSE)

rm_cols <- c(rm_cols,which(var_valid$missing_count>0))

# 3. Character columns, with lots of blank values
col_classes <- as.data.frame(lapply(raw_training,class))
chr_cols <- grep("character",t(col_classes))

# 4. Identify the columns to remove, and remove
rm_cols <- union(rm_cols,chr_cols[(chr_cols!=which(names(raw_training)=="classe"))])
training <- raw_training[,-rm_cols]
testing <- raw_testing[,-rm_cols]

# 5. Convert the classifier variable to factor
training$classe <- as.factor(training$classe)

####### Part 2. Establish baseline #############
# In this section, a random forest model is created for classification of the 5 categories of observations.
# The default settings are used for training, with 5 fold cross validation to assess performance of the model.

# Function to estimate the performance of the model
get_performance <- function(training,training_folds,k=5) {
    
    n_vars <- ncol(training) -1
    var_imp <- data.frame(temp=rep(NA,n_vars),stringsAsFactors=FALSE)
    model_vars <- data.frame(i=integer(),nvars=integer(),m_try=integer(),n_tree=integer(),
                             in_error=numeric(),out_error=numeric())
    for (i in 1:k) {
        print(paste("Run : ",i,"Training Model"))
        trainx <- training[-training_folds[[i]],]
        testx <- training[training_folds[[i]],]
        
        model_fit <- randomForest(x=trainx[,-(n_vars+1)],y=trainx[,"classe"])
        print("Done")
        pred_train <- predict(model_fit,trainx)
        in_error <- 1- sum(pred_train==trainx$classe)/nrow(trainx)
        pred_test <- predict(model_fit,testx)
        out_error <- 1 - sum(pred_test==testx$classe)/nrow(testx)
        model_vars[i,] <- c(i,n_vars,model_toy$mtry,model_toy$ntree,in_error,out_error)
        temp <- paste0("var",i)
        var_imp[,temp] <- rownames(model_toy$importance)
        temp <- paste0("rank",i)
        var_imp[,temp] <- model_toy$importance
        
    }
    
    var_imp$weights <- rowSums(var_imp[,c(3,5,7,9,11)])
    var_imp <- var_imp[,c("var1","weights")]
    var_imp <- var_imp[order(var_imp$weights,decreasing=TRUE),]
    model_perf <- mean(model_vars$out_error)
    
    return(list(model_perf=model_perf,var_imp=var_imp,model_vars=model_vars,model_fit=model_fit))
}

training_folds <- createFolds(training$classe,k=5)
baseline <- get_performance(training,5)
baseline$model_perf

## Check how many variables can be removed without impacting peformanca

model_vars <- data.frame(i=integer(),nvars=integer(),m_try=integer(),n_tree=integer(),
                         in_error=numeric(),out_error=numeric())

var_ranks <- list()
var_imp <- data.frame(temp=rep(NA,52),stringsAsFactors=FALSE)

for (i in 1:5) {
    n_vars <- dim(training)[2]-1
    trainx <- training[training_folds[[i]],]
    testx <- training[training_folds[[sample(c(1:5)[-i],1)]],]
    model_fit2 <- randomForest(x=trainx[,-53],y=trainx[,"classe"])
    pred_train <- predict(model_fit2,trainx)
    in_error <- 1- sum(pred_train==trainx$classe)/nrow(trainx)
    pred_test <- predict(model_fit2,testx)
    out_error <- 1 - sum(pred_test==testx$classe)/nrow(testx)
    model_vars[i,] <- c(i,n_vars,model_fit2$mtry,model_fit2$ntree,in_error,out_error)
    temp <- paste0("var",i)
    var_imp[,temp] <- rownames(model_fit$importance)
    temp <- paste0("rank",i)
    var_imp[,temp] <- model_fit2$importance
}

var_imp$weights <- rowSums(var_imp[,c(3,5,7,9,11)])
var_ranks <- var_imp[,c("var1","weights")]
var_ranks <- var_ranks[order(var_ranks$weights,decreasing=TRUE),]

use_vars <- c(var_ranks[1:45,1],"classe")

model_perf <- data.frame(n_vars=integer(),perf=numeric())
for (i in 1:10) {
    n_vars <- i*5
    use_vars <- c(var_ranks[1:n_vars,1],"classe")
    test_model <- get_performance(training[,use_vars],5)
    model_perf[i,] <- c(i*5,test_model$model_perf)
}

# Variable Selection Plot
model_fit <- randomForest(x=training[,-53],y=training[,53])
png(filename="Fig1_VariableImportance.png",width=800,height=650)
varImpPlot(model_fit,main="Variable Importance")
dev.off()

####### Part 3. Create Model #############
# Use only 20 variables for training the model
use_vars <- c(var_ranks[1:20,1],"classe")

# Fit the model
train_x <- training[,use_vars]
model_fit <- randomForest(x=train_x[,-21],y=train_x[,"classe"])

# Plot the classification error rates
my_colors <- c("black","blue","green","red","yellow","salmon")
png(filename="Fig2_ClassErrorRates.png",width=800,height=650)
plot(model_fit,col=my_colors,lwd=2,main="Classification Error Rates")
legend(x=400,y=0.3,c("All","A","B","C","D","E"),fill=my_colors)
dev.off()

# Create the test set
test_y <- testing[,c(use_vars[1:20],"problem_id")]

# Make the prediction
pred_test <- predict(model_fit,test_y)

# Submit

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

##### Part 4. Additional parameter exploration - Not used in writeup     #########
# 1. Identify and select useful features - we will use a small subset of the data,
# primarily because of the time it takes to train the model on the complete dataset
# Also, the train function from the "caret" package takes an inordinately long time,
# so we'll use the randomForest function from the randomForest package, which seems
# to be much faster.

# create toy training and test datasets
in_train <- createDataPartition(y=training$classe,p=0.2,list=FALSE)
train_toy <- training[in_train,]

# Now create a toy test set 
in_test <- createDataPartition(y=training$classe[-in_train],p=0.1,list=FALSE)
test_toy <- training[-in_train,][in_test,]

# Create the model with default parameters
model_toy <- randomForest(x=train_toy[,-53],y=train_toy[,53])

# Let's just check how the model does on the training set itself
predtrain_toy <- predict(model_toy,train_toy)
confusionMatrix(train_toy$classe,predtrain_toy)

# Now let's run it on the toy test set
predtest_toy <- predict(model_toy,test_toy)
confustionMatrix(test_toy$classe,predtest_toy)

# Find number of variables to sample
model_mtry <- data.frame(i=integer(),m_try=integer(),model_mtry=integer(),in_error=numeric(),out_error=numeric())

i=1
for (i in 1:26) {
    m_try <- i*2 
    model_toy <- randomForest(x=train_toy[,-53],y=train_toy[,53],mtry=m_try,ntree=100)
    predtrain_toy <- predict(model_toy,train_toy)
    in_error <- 1- sum(predtrain_toy==train_toy$classe)/nrow(train_toy)
    predtest_toy <- predict(model_toy,test_toy)
    out_error <- 1 - sum(predtest_toy==test_toy$classe)/nrow(test_toy)
    model_mtry[i,] <- c(i,m_try,model_toy$mtry,in_error,out_error)
    #model_data <- rbind(model_data,c(i=i,m_try=m_try,in_error=in_error,out_error=out_error))
}

plot(x=model_data$m_try,y=model_data$out_error,main="Classifiction Error",xlab="Num. vars sampled",ylab="Error")


# find number of trees to grow
model_tree <- data.frame(i=integer(),m_try=integer(),n_tree=integer(),model_ntree=integer(),
                            in_error=numeric(),out_error=numeric())
for (i in 1:10) {
    m_try <- 10
    n_tree <- 50+i*50
    model_toy <- randomForest(x=train_toy[,-53],y=train_toy[,53],mtry=m_try,ntree=n_tree)
    predtrain_toy <- predict(model_toy,train_toy)
    in_error <- 1- sum(predtrain_toy==train_toy$classe)/nrow(train_toy)
    predtest_toy <- predict(model_toy,test_toy)
    out_error <- 1 - sum(predtest_toy==test_toy$classe)/nrow(test_toy)
    model_tree[i,] <- c(i,m_try,n_tree,model_toy$ntree,in_error,out_error)
    #model_data <- rbind(model_data,c(i=i,m_try=m_try,in_error=in_error,out_error=out_error))
}

plot(x=model_tree$model_ntree,y=model_tree$out_error,main="Classifiction Error",xlab="Num. vars sampled",ylab="Error")


# Let's understand which of the variables are important
n_try <- 10
n_tree <- 350
model_toy <- randomForest(x=train_toy[,-53],y=train_toy[,53],mtry=m_try,ntree=n_tree)
my_colors <- c("black","blue","green","red","yellow","salmon")
plot(model_toy,col=my_colors,lwd=2,main="Classification Error Rates")
legend(x=250,y=0.20,c("All","A","B","C","D","E"),fill=my_colors,cex=0.7)
varImpPlot(model_toy,main="Variable Importance")

rank_vars <- varImp(model_toy)[[1]]
rank_vars$vars <- rownames(rank_vars)
rownames(rank_vars) <- NULL
rank_vars <-rank_vars[,c(2,1)]
rank_vars <- rank_vars[order(rank_vars$Overall,decreasing=TRUE),]
imp_vars <- rank_vars[rank_vars$Overall>10,]

rank_vars <- data.frame(vars=rownames(model_toy$importance),stringsAsFactors=FALSE)
rank_vars$MeanDecreaseGini <- model_toy$importance
rank_vars <- rank_vars[order(rank_vars$MeanDecreaseGini,decreasing=TRUE),]
rownames(rank_vars) <- NULL

# based on the simulations above, these seem to be reasonable assumptions
m_try <- 10
n_tree <- 350


