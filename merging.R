# Title: HarvardX Data Science: Capstone Project 2 Parkinsons with ensemble methods
# Jan Thomsen
# Date last edited: 05/24/2021
# Reference: Edx Data Science course notes 
# and textbook "Introduction to Data Science Data Analysis and Prediction Algorithms with R" By Prof. Rafael A. Irizarry
# Other citations is in the pdf file

# Install all needed libraries if it is not present
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(randomForest)) install.packages("randomForest")
if(!require(reshape2)) install.packages("reshape2")
if(!require(pROC)) install.packages("pROC")
if(!require(mvnormtest)) install.packages("mvnormtest")
if(!require(tibble)) install.packages("tibble")
if(!require(corrplot)) install.packages("corrplot")
if(!require(matrixStats)) install.packages("matrixStats")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(rpart)) install.packages("rpart")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(lattice)) install.packages("lattice")
if(!require(funModeling)) install.packages("funModeling")
if(!require(stringr)) install.packages("stringr")
if(!require(caTools)) install.packages("caTools")
if(!require(alookr)) install.packages("alookr")
if(!require(AppliedPredictiveModeling)) install.packages("AppliedPredictiveModeling")
if(!require(readr)) install.packages("readr")
if(!require(gt)) install.packages("gt")
if(!require(RSQLite)) install.packages("RSQLite")
if(!require(knitr)) install.packages("knitr")
if(!require(RhpcBLASctl)) install.packages("RhpcBLASctl")
if(!require(formattable)) install.packages("formattable")
if(!require(ggpubr)) install.packages("ggpubr")
if(!require(RColorBrewer)) install.packages("RColorBrewer")
if(!require(ModelMetrics)) install.packages("ModelMetrics")
if(!require(mmtable2)) install.packages("mmtable2")
if(!require(GGally)) install.packages("GGally")
if(!require(mlbench)) install.packages("mlbench")
if(!require(plyr)) install.packages("plyr")

#Loading the packages
library(plyr)
library(mlbench)
library(GGally)
library(mmtable2)
library(ModelMetrics)
library(RColorBrewer)
library(formattable)
library(ggpubr)
library(RhpcBLASctl)
library(knitr)
library(gt)
library(RSQLite)
library(AppliedPredictiveModeling)
library(readr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(caret)
library(dplyr)
library(gbm)
library(tibble)
library(reshape2)
library(randomForest)
library(corrplot)
library(matrixStats)
library(mvnormtest)
library(pROC)
library(e1071)
library(class)
library(rpart)
library(rpart.plot)
library(lattice)
library(funModeling)
library(stringr)
library(caTools)
library(alookr)

#Read the data file
parkinsons <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data", header = TRUE)
#parkinsons <- read.csv("~/Desktop/telemonitoring_parkinsons_updrs.csv", header = TRUE)
glimpse(parkinsons, width = getOption("width"))

#designing the multivariate normality analysis
x <- parkinsons
cm <- colMeans(x)
S <- cov(x)
d <- apply(x, 1, function(x) t(x - cm) %*% solve(S) %*% (x - cm))
# Chi-Square plot:
plot(qchisq((1:nrow(x) - 1/2) / nrow(x), df = ncol(x)), 
     sort(d),
     xlab = expression(paste(chi[22]^2, 
                             " Quantile")), 
     ylab = "Ordered distances")
abline(a = 0, b = 1)

#checking if any data is missing
missing <- apply(parkinsons, 2, function(x) 
  round(100 * (length(which(is.na(x))))/length(x) , digits = 1))
knitr::kable(missing)

#checking correlation
corrplot(cor(parkinsons), type="full", method ="color", title = "Parkinson correlation plot", mar=c(0,0,1,0), tl.cex= 0.8, outline= T, tl.col="indianred4")

# overview of data
summary(parkinsons[,-3])

#proportions of measurements
# Distribution of the sex column
df <- as.factor(parkinsons$sex)
parkinsons$sex <- factor(parkinsons$sex, 
                 levels=c(0,1), 
                 labels=c("male","female"))

options(repr.plot.width=4, repr.plot.height=4)
ggplot(parkinsons, aes(x=sex))+geom_bar(fill="red",alpha=1)+theme_bw()+labs(title="Distribution of gender")


#average severity and testtime distributed on gender
parkinsons$sex <- as.factor(parkinsons$sex)
data_wrangled <- parkinsons %>%
  group_by(parkinsons$sex) %>%
  summarise(avg = mean(total_UPDRS), sd = sd(total_UPDRS))

print(data_wrangled)

# Severity and sex plot
parkinsons$sex <- as.factor(parkinsons$sex)
mu <- ddply(parkinsons, "sex", summarise, grp.mean=mean(total_UPDRS))
head(mu)

p<-ggplot(parkinsons, aes(x=total_UPDRS, fill=sex)) +
  geom_density(alpha=0.5)+
  geom_vline(data=mu, aes(xintercept=grp.mean, color=sex),
             linetype="dashed", size=1) + ggtitle("Densityplot of total_UPDRS and sex") + theme(plot.title = element_text(color="black", size=18, face="bold.italic"),
axis.title.x = element_text(color="black", size=16, face="bold"),
axis.title.y = element_text(color="black", size=16, face="bold")
)



#facet density plot  of the genders
attach(parkinsons)
parkinsons$sex=as.factor(parkinsons$sex)
park.m=melt(parkinsons[,-1], id.vars = "sex")

p <- ggplot(data = park.m, aes(x=value)) +  geom_density(aes(fill=sex), alpha = 0.5)
p <- p + facet_wrap( ~ variable, scales = "free") + ggtitle("Densityplots on dataset predictors") + theme(
plot.title = element_text(color="blue", size=20, face="bold.italic"),
           axis.text.x = element_text( size = 12 ),
           axis.title = element_text( size = 12, face = "bold" ),
           legend.position="right",
           strip.text = element_text(size = 12))
p


#Average and variance plot on gender
attach(parkinsons)
parkinsons$sex=as.factor(parkinsons$sex)
park.m=melt(parkinsons[,-1], id.vars = "sex")

p <- ggplot(data = park.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=sex))
p + facet_wrap( ~ variable, scales="free") + ggtitle("Gender means and variance on predictors") + theme(
plot.title = element_text(color="black", size=20, face="bold.italic"),
           axis.text.x = element_text( size = 12 ),
           axis.title = element_text( size = 12, face = "bold" ),
           legend.position="right",
           strip.text = element_text(size = 12))


#Featureplot on variables on gender
transparentTheme(trans = .4)
caret::featurePlot(x = parkinsons[, 4:8], 
            y = parkinsons$sex, 
            plot = "ellipse", jitter = TRUE,
            auto.key = list(columns = 5))
#Title: Gender Scatterplot on predictors 

# boxplot for total UPDRS by different subjects
fill <- "blue"
line <- "black"
ggplot(parkinsons, aes(x =as.factor(parkinsons$subject.), y =parkinsons$total_UPDRS)) +
        geom_boxplot(fill = fill, colour = line) +
        scale_y_continuous(name = "total UPDRS",
                           breaks = seq(0, 60, 2),
                           limits=c(5, 60)) +
        scale_x_discrete(name = "subject") +
        ggtitle("Boxplot of total_UPDRS and subject")+ theme(
plot.title = element_text(color="black", size=20, face="bold.italic"),
axis.title.x = element_text(color="black", size=14, face="bold"),
axis.title.y = element_text(color="black", size=14, face="bold")
)

#Severity and age plot
fill <- "red"
line <- "black"
ggplot(parkinsons, aes(x =as.factor(parkinsons$age), y =parkinsons$total_UPDRS)) +
        geom_boxplot(fill = fill, colour = line) +
        scale_y_continuous(name = "total UPDRS",
                           breaks = seq(10, 60, 1),
                           limits=c(10, 60)) +
        scale_x_discrete(name = "age") +
        ggtitle("Boxplot of total_UPDRS and age") +theme(
plot.title = element_text(color="black", size=24, face="bold.italic"),
axis.title.x = element_text(color="black", size=16, face="bold"),
axis.title.y = element_text(color="black", size=16, face="bold")
)

# Remove correlated variables x>0.9
parkinsons$sex <- as.factor(parkinsons$sex)
parkinsons$age <- as.numeric(parkinsons$age)
parkinsons <- treatment_corr(parkinsons) #, corr_thres = 0.9, treat = FALSE)

# Number of columns after removing correlated variables
ncol(parkinsons)
names(parkinsons)


#Excluding variable subject
parkinsons <- parkinsons[-1]
str(parkinsons)

#Data Partition
data2 <- parkinsons
set.seed(18)
pd_idx = sample(1:nrow(data2), nrow(data2) / 2)
pd_trn = data2[pd_idx,]
pd_tst = data2[-pd_idx,]


#Designing the tree
pd_tree = rpart(total_UPDRS ~ ., data = pd_trn)

#Predicting the tree
pd_tree_tst_pred = predict(pd_tree, newdata = pd_tst)


rpart.plot(pd_tree)

#plotting predicted and actual single tree
plot(pd_tree_tst_pred, pd_tst$total_UPDRS, 
     xlab = "Predicted", ylab = "Actual", 
     main = "Predicted vs Actual: Single Tree, Test Data",
     col = "blue", pch = 20)
grid()
abline(0, 1, col = "red", lwd = 2)

#calculating RMSE of rpart model
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
(tree_tst_rmse = calc_rmse(pd_tree_tst_pred, pd_tst$total_UPDRS))


#Designing the lm model
pd_lm = lm(total_UPDRS ~ ., data = pd_trn)

pd_lm_tst_pred = predict(pd_lm, newdata = pd_tst)

#plotting the lm model actual vs. predicted
plot(pd_lm_tst_pred, pd_tst$total_UPDRS,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Linear Model, Test Data",
     col = "blue", pch = 20)
grid()
abline(0, 1, col = "red", lwd = 2)
(lm_tst_rmse = calc_rmse(pd_lm_tst_pred, pd_tst$total_UPDRS))


#Bagging the RF algorithm
pd_bag = randomForest(total_UPDRS ~ ., data = pd_trn, mtry = 10, 
                          importance = TRUE, ntrees = 500)
pd_bag

#plotting the bagged model actual vs. predicted
pd_bag_tst_pred = predict(pd_bag, newdata = pd_tst)
plot(pd_bag_tst_pred,pd_tst$total_UPDRS,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Bagged Model, Test Data",
     col = "blue", pch = 20)
grid()
abline(0, 1, col = "red", lwd = 2)

(bag_tst_rmse = calc_rmse(pd_bag_tst_pred, pd_tst$total_UPDRS))

#plotting the error of the trees
plot(pd_bag, col = "blue", lwd = 2, main = "Bagged Trees: Error vs Number of Trees")
grid()

#Designing the Random Forest model
pd_forest = randomForest(total_UPDRS ~ ., data = pd_trn, mtry = 4, 
                             importance = TRUE, ntrees = 500)
pd_forest



#Checking the most important variables
varImpPlot(pd_forest,
           sort = T,
           n.var = 5,
           main = "RF Top 5 - Variable Importance")

#Random forest plot actual versus predicted values
pd_forest_tst_pred = predict(pd_forest, newdata = pd_tst)
plot(pd_forest_tst_pred, pd_tst$total_UPDRS,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Random Forest, Test Data",
     col = "blue", pch = 20)
grid()
abline(0, 1, col = "red", lwd = 2)

#Calculating the RMSE of RF
(forest_tst_rmse = calc_rmse(pd_forest_tst_pred, pd_tst$total_UPDRS))


pd_forest_trn_pred = predict(pd_forest, newdata = pd_trn)
forest_trn_rmse = calc_rmse(pd_forest_trn_pred, pd_trn$total_UPDRS)
forest_oob_rmse = calc_rmse(pd_forest$predicted, pd_trn$total_UPDRS)



pd_boost = gbm(total_UPDRS ~ ., data = pd_trn, distribution = "gaussian", 
                    n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
pd_boost

tibble::as_tibble(summary(pd_boost))

par(mfrow = c(1, 3))
plot(pd_boost, i = "age", col = "blue", lwd = 2)

plot(pd_boost, i = "DFA", col = "blue", lwd = 2)

plot(pd_boost, i = "HNR", col = "blue", lwd = 2)



pd_boost_tst_pred = predict(pd_boost, newdata = pd_tst, n.trees = 5000)

(boost_tst_rmse = calc_rmse(pd_boost_tst_pred, pd_tst$total_UPDRS))



plot(pd_boost_tst_pred, pd_tst$total_UPDRS,
     xlab = "Predicted", ylab = "Actual", 
     main = "Predicted vs Actual: Boosted Model, Test Data",
     col = "blue", pch = 20)
grid()
abline(0, 1, col = "red", lwd = 2)

#Resume of the testerrors RMSE 
(
  pd_rmse = data.frame(
  Model = c("Single Tree", "Linear Model", "Bagging",  "Random Forest",  "Boosting"),
  TestError = c(tree_tst_rmse, lm_tst_rmse, bag_tst_rmse, forest_tst_rmse, boost_tst_rmse)
)
)
print(pd_rmse)


#setting the accumulation function
calc_acc = function(actual, predicted) {
  mean(actual == predicted)
}

#designing the rpart tree
pdc_tree = rpart(sex ~ ., data = pd_trn)

#plotting the tree
rpart.plot(pdc_tree)


#Predicting the tree
pdc_tree_tst_pred = predict(pdc_tree, pd_tst, type = "class")
table(predicted = pdc_tree_tst_pred, actual = pd_tst$sex)

#Calculating the accuracy
(tree_tst_acc = calc_acc(predicted = pdc_tree_tst_pred, actual = pd_tst$sex))

#Designing the glm
pdc_glm = glm(sex ~ ., data = pd_trn, family = "binomial")
pdc_glm
pdc_glm_tst_pred = ifelse(predict(pdc_glm, pd_tst, "response") > 0.5, 
                           "female", "male")
table(predicted = pdc_glm_tst_pred, actual = pd_tst$sex)


(glm_tst_acc = calc_acc(predicted = pdc_glm_tst_pred, actual = pd_tst$sex))

# Bagging calculation
pdc_bag = randomForest(sex ~ ., data = pd_trn, mtry = 10, 
                        importance = TRUE, ntrees = 500)
pdc_bag

#Predicting the bag
pdc_bag_tst_pred = predict(pdc_bag, newdata = pd_tst)
table(predicted = pdc_bag_tst_pred, actual = pd_tst$sex)

(bag_tst_acc = calc_acc(predicted = pdc_bag_tst_pred, actual = pd_tst$sex))


# Designing the RF
pdc_forest = randomForest(sex ~ ., data = pd_trn, mtry = 3, importance = TRUE, ntrees = 500)
pdc_forest

#Predicting the forest
pdc_forest_tst_perd = predict(pdc_forest, newdata = pd_tst)
table(predicted = pdc_forest_tst_perd, actual = pd_tst$sex)

(forest_tst_acc = calc_acc(predicted = pdc_forest_tst_perd, actual = pd_tst$sex))


# A boosted gbm model
pdc_trn_mod = pd_trn
pdc_trn_mod$sex = as.numeric(ifelse(pdc_trn_mod$sex == "female", "0", "1"))

pdc_boost = gbm(sex ~ ., data = pdc_trn_mod, distribution = "bernoulli", 
                 n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
pdc_boost

#Predicting the boosted gbm
pdc_boost_tst_pred = ifelse(predict(pdc_boost, pd_tst, n.trees = 5000, "response") > 0.5, 
                             "male", "female")
table(predicted = pdc_boost_tst_pred, actual = pd_tst$sex)

(boost_tst_acc = calc_acc(predicted = pdc_boost_tst_pred, actual = pd_tst$sex))


tibble::as_tibble(summary(pdc_boost))


# Collecting the accuracies
(pdc_acc = data.frame(
  Model = c("Single Tree", "Logistic Regression", "Bagging",  "Random Forest",  "Boosting"),
  TestAccuracy = c(tree_tst_acc, glm_tst_acc, bag_tst_acc, forest_tst_acc, boost_tst_acc)
)
)

print(pdc_acc)

#Tuning
# tuning with caret
oob = trainControl(method = "oob")
cv_5 = trainControl(method = "cv", number = 5)

dim(pd_trn)

rf_grid =  expand.grid(mtry = 1:10)
library(caret)
set.seed(825)
pdc_rf_tune = train(sex ~ ., data = pd_trn,
                     method = "rf",
                     trControl = oob,
                     verbose = FALSE,
                     tuneGrid = rf_grid)
pdc_rf_tune

best_tun_mod <- calc_acc(predict(pdc_rf_tune, pd_tst), pd_tst$sex)
(tuned_tst_acc = calc_acc(predict(pdc_rf_tune, pd_tst), pd_tst$sex))
pdc_rf_tune$bestTune

#Boosting for optimization with the expand.grid function
gbm_grid =  expand.grid(interaction.depth = 1:5,
                        n.trees = (1:6) * 500,
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 10)


pdc_gbm_tune = train(sex ~ ., data = pd_trn,
                      method = "gbm",
                      trControl = cv_5,
                      verbose = FALSE,
                      tuneGrid = gbm_grid)


calc_acc(predict(pdc_gbm_tune, pd_tst), pd_tst$sex)


(boost2_tst_acc = calc_acc(predict(pdc_gbm_tune, pd_tst), pd_tst$sex))

#calculating the accuracy
(boost_tst_acc = calc_acc(predicted = pdc_boost_tst_pred, actual = pd_tst$sex))

#plotting the tuned tree
plot(pdc_gbm_tune)

print(pdc_gbm_tune$bestTune)

# Calculating the ROC boosted tuned model
test_prob = predict(pdc_glm, newdata = pd_tst, type = "response")
test_roc = roc(pd_tst$sex ~ test_prob, plot = TRUE, print.auc = TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage",  col="#377eb8", lwd=4)


#Resume of the results
(pdc_acc = data.frame(
  Model = c("Single Tree", "Logistic Regression", "Bagging",  "Random Forest",  "Boosting", "Tuned model", "Boosted tuned Model"),
  TestAccuracy = c(tree_tst_acc, glm_tst_acc, bag_tst_acc, forest_tst_acc, boost_tst_acc, tuned_tst_acc, boost2_tst_acc)
)
)

print(pdc_acc)


## knitr::purl("merging.Rmd", documentation = 0)

#setting training and test data
set.seed(42)
sim_trn = mlbench.circle(n = 1000, d = 2)
sim_trn = data.frame(sim_trn$x, class = as.factor(sim_trn$classes))
sim_tst = mlbench.circle(n = 1000, d = 2)
sim_tst = data.frame(sim_tst$x, class = as.factor(sim_tst$classes))

#Defining true positives and plotting
sim_trn_col = ifelse(sim_trn$class == 1, "red", "blue")
plot(sim_trn$X1, sim_trn$X2, col = sim_trn_col,
     xlab = "X1", ylab = "X2", main = "Simulated Training Data", pch = 20)
grid()

cv_5 = trainControl(method = "cv", number = 5)
oob  = trainControl(method = "oob")

sim_tree_cv = train(class ~ .,
                    data = sim_trn,
                    trControl = cv_5,
                    method = "rpart")

#Print tree
rpart.plot(sim_tree_cv$finalModel)


rf_grid = expand.grid(mtry = c(1, 2))
sim_rf_oob = train(class ~ .,
                   data = sim_trn,
                   trControl = oob,
                   tuneGrid = rf_grid)

gbm_grid =  expand.grid(interaction.depth = 1:5,
                        n.trees = (1:6) * 500,
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 10)

sim_gbm_cv = train(class ~ ., 
                   data = sim_trn,
                   method = "gbm",
                   trControl = cv_5,
                   verbose = FALSE,
                   tuneGrid = gbm_grid)
#plotting the grid
plot_grid = expand.grid(
  X1 = seq(min(sim_tst$X1) - 1, max(sim_tst$X1) + 1, by = 0.01),
  X2 = seq(min(sim_tst$X2) - 1, max(sim_tst$X2) + 1, by = 0.01)
)

tree_pred = predict(sim_tree_cv, plot_grid)
rf_pred   = predict(sim_rf_oob, plot_grid)
gbm_pred  = predict(sim_gbm_cv, plot_grid)

tree_col = ifelse(tree_pred == 1, "red", "blue")
rf_col   = ifelse(rf_pred == 1, "red", "blue")
gbm_col  = ifelse(gbm_pred == 1, "red", "blue")


par(mfrow = c(1, 3))
plot(plot_grid$X1, plot_grid$X2, col = tree_col,
     xlab = "X1", ylab = "X2", pch = 20, main = "Single Tree",
     xlim = c(-1, 1), ylim = c(-1, 1))
plot(plot_grid$X1, plot_grid$X2, col = rf_col,
     xlab = "X1", ylab = "X2", pch = 20, main = "Random Forest",
     xlim = c(-1, 1), ylim = c(-1, 1))
plot(plot_grid$X1, plot_grid$X2, col = gbm_col,
     xlab = "X1", ylab = "X2", pch = 20, main = "Boosted Trees",
     xlim = c(-1, 1), ylim = c(-1, 1))



# CLEAN UP 

# Clear environment
rm(list = ls()) 

# Clear console
cat("\014")  #ctrl+L

