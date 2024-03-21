## linear model
library(caret)
library(car)
library(gvlma)
set.seed(20230409)
data <- read.csv("train.csv", header = TRUE)
# 全模型回归
fit <- lm(log(Delivery) ~ AvePrice + HousePrice + Nightlight + PopDens + KernelOffi 
          + RoadDens + BurgerNumber + NumberRatio + AveDistance + BurgerDeliveryElse + DeliveryRatio, data = data)
summary(fit)

# 模型诊断
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
fit2 <- lm(log(Delivery) ~ AvePrice + Nightlight + BurgerNumber + NumberRatio + BurgerDeliveryElse, data = data)
summary(fit2)

#预测
newdata <- read.csv("test.csv", header = TRUE)
newdata <- data.frame(newdata[, c(4:14)])
Delivery_hat <- predict(fit2, newdata = newdata)
write.csv(Delivery_hat, "Delivery_hat.csv", row.names = TRUE)

