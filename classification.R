
############### xgboost分类 ######################
## 多重插补
library(mice)
library(dplyr)
library(magrittr)

data <- read.csv("hamdata.csv",row.names = 1)
data %>% 
  filter(if_any(everything(), is.na))
ic(data) 
Nadata <- data %>% 
  dplyr::select_if(~any(is.na(.)))
md.pattern(Nadata,rotate.names = TRUE)
data %>% 
  select_if(~any(is.na(.))) %>%  
  summarise(across(where(is.numeric),list(
    n = ~ sum(is.na(.)),
    prop = ~ round(sum(is.na(.)) / length(.),3)
  )))
count(data,HousePrice)

names(Nadata)
tempData <- mice(Nadata,
                 m=20, 
                 maxit=5, 
                 method = c('cart','cart','cart'), 
                 seed = 12345 )
summary(tempData)
tempData$meth 
stripplot(x= tempData)

completedData <- complete(tempData,3) 
data[,c(3,4,6)] <- completedData[,c(1,2,3)]

write.csv(data, "hamdata_new.csv", row.names = TRUE)

## 分类
library(xgboost)
library(caret)
library(Matrix)
library(pROC)
set.seed(20230409)
setwd("C:/Users/HP/Desktop")
data <- read.csv("hamdata_new.csv", header = TRUE)

trainlist <- createDataPartition(data$Burger, p = 0.8, list = FALSE)
trainset <- data[trainlist, ]
testset <- data[-trainlist, ]

traindata1 <- data.matrix(trainset[, c(3,4,5,6,7,8)])
traindata2 <- Matrix(traindata1, sparse = T)
train_y <- trainset[, 9]
traindata <- list(data = traindata2, label = train_y)
dtrain <- xgb.DMatrix(data = traindata$data, label = traindata$label)

testset1 <- data.matrix(testset[, c(3,4,5,6,7,8)])
testset2 <- Matrix(testset1, sparse = T)
test_y <- testset[, 9]
testset <- list(data = testset2, label = test_y)
dtest <- xgb.DMatrix(data = testset$data, label = testset$label)

fset1 <- data.matrix(data[, c(3,4,5,6,7,8)])
fset2 <- Matrix(fset1, sparse = T)
f_y <- data[, 9]
fset <- list(data = fset2, label = f_y)
f <- xgb.DMatrix(data = fset$data, label = fset$label)

weight = sum(train_y == 0) / sum(train_y == 1)
model_xgb <- xgboost(data = dtrain, booster = 'gbtree', maxdepth = 3, subsample = 0.9, colsample_bytree=0.9, 
                     eta = 0.5, objective = 'multi:softmax', 
                     num_class = 3, nrounds = 20, scale_pos_weight = weight)
pre <- predict(model_xgb, newdata = dtest)

xgb.cf <- caret::confusionMatrix(as.factor(pre), as.factor(test_y))
xgb.cf

xgboost_roc <- roc(testset$label, as.numeric(pre))
plot(xgboost_roc, print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2))

#tune
# 
# param_space <- list(
#   eta = 0.1,
#   max_depth = c(3, 4, 5),
#   subsample = c(0.7, 0.8, 0.9),
#   colsample_bytree = c(0.7, 0.8, 0.9)
# )
# cv_params <- list(
#   nfold = 5,
#   metrics = "auc",
#   early_stopping_rounds = 10,
#   stratified = TRUE
# )
# xgb.cv <- xgb.cv(
#   params = param_space, 
#   data = dtrain, 
#   nrounds = 1000, 
#   nfold = 5,
#   stratified = cv_params$stratified, 
#   metrics = cv_params$metrics,
#   early_stopping_rounds = cv_params$early_stopping_rounds,
#   maximize = TRUE,
#   verbose = TRUE
# )
# 
# print(xgb.cv)
# 
# best_param <- xgb.cv$evaluation_log[xgb.cv$best_iteration, ]
# print(best_param)












############################## linear model ####################################
library(caret)
library(car)
library(gvlma)
set.seed(20230409)
setwd("C:/Users/HP/Desktop")
data <- read.csv("train2.csv", header = TRUE)
# 全模型回归
fit <- lm(Delivery ~ AvePrice + HousePrice + Nightlight + PopDens + KernelOffi 
          + RoadDens + BurgerNumber + NumberRatio + AveDistance + BurgerDeliveryElse + DeliveryRatio, data = data)
summary(fit)

#### 模型诊断
#正态性
qqPlot(fit)
#自相关性
durbinWatsonTest(fit)
#线性
crPlots(fit)

#同方差性
ncvTest(fit)
#线性模型假设的综合验证
gvmodel <- gvlma(fit)
summary(gvmodel)
#多重共线性，虽然有多重共线性，但如果目的是预测而不是对变量进行解释，则影响不大
vif(fit)
sqrt(vif(fit))


##向后回归
library(MASS)
stepAIC(fit, direction = "backward")
#用AIC筛选后的变量进行回归
fit2 <- lm(Delivery ~ AvePrice + Nightlight + KernelOffi + BurgerNumber + NumberRatio + BurgerDeliveryElse, data = data)
summary(fit2)



#预测
newdata <- read.csv("test.csv", header = TRUE)
newdata <- data.frame(newdata[, c(4:14)])
Delivery_hat <- predict(fit2, newdata = newdata)
write.csv(Delivery_hat, "Delivery_hat_test.csv", row.names = TRUE)






################## xgboost回归 ###################
library(xgboost)
library(caret)
set.seed(20230411)
x <- as.matrix(data[, 4:14])
y <- data[, 15]

train_indices <- createDataPartition(y, times = 1, p = 0.7, list = FALSE)
train_x <- x[train_indices, ]
train_y <- y[train_indices]
test_x <- x[-train_indices, ]
test_y <- y[-train_indices]

# 训练xgboost模型
model <- xgboost(data = train_x, label = train_y, objective = "reg:squarederror",
                 nthread = 2, max.depth = 6, nrounds = 100, eta = 0.1)

model2 <- xgboost(data = train_x, label = train_y, objective = "multi:softmax",
                 nthread = 2, max.depth = 6, nrounds = 100, eta = 0.1, num_class = 19)

model_xgb <- xgboost(data = dtrain, booster = 'gbtree', maxdepth = 3, subsample = 0.9, colsample_bytree=0.9, 
                     eta = 0.5, objective = 'multi:softmax', 
                     num_class = 3, nrounds = 20, scale_pos_weight = weight)


# 使用训练好的xgboost模型进行预测
pred_y2 <- predict(model2, newdata = test_x)

# 模型评估，例如计算均方误差（Mean Squared Error）
mse2 <- mean((pred_y2 - test_y)^2)

################# order logistic #################
library(MASS)
setwd("C:/Users/HP/Desktop")
data <- read.csv("train2.csv", header = TRUE)
data$response <- ordered(data$order)
fit3 <- polr( response ~ AvePrice + HousePrice + Nightlight + PopDens + KernelOffi 
                     + RoadDens + BurgerNumber + NumberRatio + AveDistance + BurgerDeliveryElse 
                     + DeliveryRatio, data = data)
drop1(fit3, test = "Chi")




############# 多元多项式回归 #############
setwd("C:/Users/HP/Desktop")
data <- read.csv("train.csv", header = TRUE)
degree <- 2
poly_features <- cbind(poly(data$AvePrice, degree, raw = TRUE), 
                       poly(data$BurgerNumber, degree, raw = TRUE), 
                       poly(data$Nightlight, degree, raw = TRUE), 
                       poly(data$NumberRatio, degree, raw = TRUE), 
                       poly(data$BurgerDeliveryElse, degree, raw = TRUE)
                       ) 
model <- lm(data$Delivery ~ poly_features)
newdata2 <-  read.csv("train.csv", header = TRUE)
pred_2333 <- predict(model, newdata = newdata2)
write.csv(pred_2333, "2333.csv", row.names = TRUE)

############# 描述性统计 ################
library(plyr)
library(ggplot2)
library(ggpubr)
library(RColorBrewer)
setwd("C:/Users/HP/Desktop")
data <- read.csv("train.csv", header = TRUE)
summary(data)

#箱线图
x = data$Nightlight
y = data$Delivery
x_median <- median(x)
x_group <- ifelse(x < x_median, "较低", "较高")

temp <- data.frame(x = x, y = y, group = x_group)

ggplot(temp, aes(x = group, y = y, fill = group)) +
  geom_boxplot() +
  labs(x = "夜晚灯光", y = "对数化汉堡店月销量") +
  theme_minimal() +
  theme(legend.position = c('none')) +
  scale_fill_manual(values = c("#FFCEAC", "#FFCEAC"))
