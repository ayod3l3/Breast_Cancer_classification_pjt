setwd("/Users/ayodeleogundele/Desktop/Uni_Sunderland/CETM72/CETM72_R")

set.seed(123)

# Load Libraries
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(pROC)
library(corrplot)
library(rpart)
library(rpart.plot)
library(DMwR2)

#Load dataset from working directory
bcdata <- read_csv("wisconsin.csv")

#Data inspection 
head(bcdata)
summary(bcdata)
str(bcdata)


#Identify columns with missing values
missing_values <- colSums(is.na(bcdata))
columns_with_missingV <- names(missing_values[missing_values > 0])

columns_with_missingV

#Handle missing values with mean
bcdata$Bare.nuclei[is.na(bcdata$Bare.nuclei)] <- mean(bcdata$Bare.nuclei, 
                                                      na.rm = TRUE)

#Check that there are no more missing values
sum_missing <- sum(colSums(is.na(bcdata)))
sum_missing

#Convert target variable to factors
bcdata$Class <- factor(bcdata$Class)

#Plotting class distribution using a bar plot
barplot(table(bcdata$Class), 
        main = "Class Distribution in bcdata",
        xlab = "Class",
        ylab = "Frequency")


#Model without feature selection
#Splitting the data into 70% train and 30% test sets
bc_index <- sample(nrow(bcdata), 0.7 * nrow(bcdata))
bctrain <- bcdata[bc_index, ]
bctest <- bcdata[-bc_index, ]

#View class distribution for train and test subset
table(bctrain$Class)
table(bctest$Class)

#Handle imbalanced in bctrain using undersampling 
#Count the class distribution in bctrain
class_distribution <- table(bctrain$Class)

#Identify class with the majority and minority class
majority_class <- names(which.max(class_distribution))
minority_class <- names(which.min(class_distribution))

#Identify the indices of majority class instances
majority_indices <- which(bctrain$Class == majority_class)

#Random sample from the majority class to match minority class instances and undersampled majority class
undersampled_majority_indices <- sample(majority_indices, sum(bctrain$Class == minority_class))
undersampled_indices <- c(which(bctrain$Class == minority_class), undersampled_majority_indices)

# Create an undersampled dataset
bctrain_undersampled <- bctrain[undersampled_indices, ]
table(bctrain_undersampled$Class)

#Perform classification using Random Forest
rf_model <- train(Class ~ ., data = bctrain_undersampled, method = "rf", 
                  trControl = trainControl(method = "cv", 
                                           number = 5, verboseIter = TRUE))

#Use the trained model to predict on the test set
rf_predictions <- predict(rf_model, newdata = bctest)

#Assess model performance on the test set
rf_confusion <- confusionMatrix(rf_predictions, bctest$Class)

# Extract other metrics
rf_precision <- rf_confusion$byClass['Precision']
rf_recall <- rf_confusion$byClass['Recall']
rf_F1 <- rf_confusion$byClass['F1']
rf_precision 
rf_recall 
rf_F1 


# Perform classification using Decision Tree
tree_model <- rpart(Class ~ ., data = bctrain_undersampled, method = "class")

# Make predictions on the test set
tree_predictions <- predict(tree_model, newdata = bctest, type = "class")

#Assess model performance and extract metrics
tree_confusion <- confusionMatrix(tree_predictions, bctest$Class)

# Extract other metrics
tree_precision <- tree_confusion$byClass['Precision']
tree_recall <- tree_confusion$byClass['Recall']
tree_F1 <- tree_confusion$byClass['F1']
tree_precision 
tree_recall 
tree_F1 

# Plot the decision tree
rpart.plot(tree_model)



#Model development with feature selection
#Correlation analysis
#Convert Class variable to numeric (assuming benign as 0 and malignant as 1)
bcdata$Class_numeric <- as.numeric(factor(bcdata$Class, levels = c("benign", "malignant"), labels = c(0, 1)))

numeric_cols <- bcdata %>% select_if(is.numeric)
corr_matrix <- cor(numeric_cols)
corrplot(corr_matrix, method = "number", tl.cex = 0.5)

#Select predictors based on correlation threshold of 0.6
selected_predictors <- names(which(corr_matrix["Class_numeric", ] > 0.6))[-1] # Exclude 'Class_numeric' itself

#Subset data with selected predictors
bcdata_selected <- bcdata[, c("Class", selected_predictors)]

#Convert target variable to factor
bcdata_selected$Class <- factor(bcdata_selected$Class)

#Splitting the data into 70% train and 30% test sets
bc_index_selected <- sample(nrow(bcdata_selected), 0.7 * nrow(bcdata_selected))
bctrain_selected <- bcdata_selected[bc_index_selected, ]
bctest_selected <- bcdata_selected[-bc_index_selected, ]

#View class distribution for train and test subset
table(bctrain_selected$Class)
table(bctest_selected$Class)

#Perform classification using Random Forest
rf_model_selected <- train(Class ~ ., data = bctrain_selected, method = "rf", 
                  trControl = trainControl(method = "cv", 
                  	number = 5, verboseIter = TRUE))

#Use the trained model to predict on the test set
rf_predictions_selected <- predict(rf_model_selected, newdata = bctest_selected)

#Assess Random Forest model performance on the test set
rf_confusion_selected <- confusionMatrix(rf_predictions_selected, bctest_selected$Class)

#Extract other metrics
rf_precision_selected <- rf_confusion_selected$byClass['Precision']
rf_recall_selected <- rf_confusion_selected$byClass['Recall']
rf_F1_selected <- rf_confusion_selected$byClass['F1']

#Perform classification using Decision Tree
tree_model_selected <- rpart(Class ~ ., data = bctrain_selected, method = "class")

#Make predictions on the test set
tree_predictions_selected <- predict(tree_model_selected, newdata = bctest_selected, type = "class")

#Assess Decision Tree model performance
tree_confusion_selected <- confusionMatrix(tree_predictions_selected, bctest_selected$Class)

#Extract other matrics
tree_precision_selected <- tree_confusion_selected$byClass['Precision']
tree_recall_selected <- tree_confusion_selected$byClass['Recall']
tree_F1_selected <- tree_confusion_selected$byClass['F1']
tree_precision_selected
tree_recall_selected
tree_F1_selected

#Plot the decision tree
rpart.plot(tree_model_selected) #this should be on tree_predictions_selected

#Compare the performance of Random Forest predictions and Decision Tree predictions with and without feature selection using ROC Curve

#Calculate ROC for Random Forest predictions without feature selection
rf_roc <- roc(ifelse(rf_predictions == "benign", 1, 0), 
                ifelse(bctest$Class == "benign", 1, 0))

#Calculate ROC for Decision Tree predictions without feature selection
tree_roc <- roc(ifelse(tree_predictions == "benign", 1, 0), 
                ifelse(bctest$Class == "benign", 1, 0))

#Calculate ROC for Random Forest predictions with feature selection
rf_roc_selected <- roc(ifelse(rf_predictions_selected == "benign", 1, 0), 
                ifelse(bctest_selected$Class == "benign", 1, 0))

#Calculate ROC for Decision Tree predictions with feature selection
tree_roc_selected <- roc(ifelse(tree_predictions_selected == "benign", 1, 0), 
                ifelse(bctest_selected$Class == "benign", 1, 0))

# Plotting ROC curves for all models
plot(rf_roc, col = "blue", legacy.axes = TRUE, print.auc = TRUE, main = "ROC Curves - Random Forest vs. Decision Tree")
lines(tree_roc, col = "red")
lines(rf_roc_selected, col = "green")
lines(tree_roc_selected, col = "orange")
legend("bottomright", legend = c("Random Forest (No Feature Selection)", "Decision Tree (No Feature Selection)", 
                                 "Random Forest (With Feature Selection)", "Decision Tree (With Feature Selection)"), 
       col = c("blue", "red", "green", "orange"), lty = 1)
