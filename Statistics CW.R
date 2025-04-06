#------------------------- Brain Response to an audio Signal -------------------------
Xdata <- read.csv("data/X.csv")
Ydata <- read.csv("data/Y.csv")
timedata <- read.csv("data/time.csv")

#-------------------------Data Inspection-------------------------
# Checking for head
head(Xdata)
head(Ydata)
head(timedata)

# Checking for the structure of data
str(Xdata)
str(Ydata)
str(timedata)

# Checking for dimensions
dim(Xdata)
dim(Ydata)
dim(timedata)

# Checking for null values
any(is.na(Xdata))
any(is.na(Ydata))
any(is.na(timedata))

summary(Xdata$x2)

#-------------------------Merging Data-------------------------
brainData <- data.frame(time = timedata$time,
                        x1 = Xdata$x1,
                        x2 = Xdata$x2,
                        y = Ydata$y)
head(brainData)

brainData$x2 <- as.factor(brainData$x2)
head(brainData)

summary(brainData$x2)

#-------------------------Task1-------------------------
# Time Series plot of input audio(x1)
plot(brainData$time,
     brainData$x1,
     type = "l",
     xlab = "Time",
     ylab = "Input Audio (x1)",
     main = "Time Series Analysis of Input Audio")

# Time Series plot of MEG signal (y)
plot(brainData$time,
     brainData$y,
     type = "l",
     xlab = "Time",
     ylab = "MEG Signal (y)",
     main = "Time Series Analysis of MEG Signal")

# Histogram of input audio (x1)
hist(brainData$x1,
     xlab = "Input Audio (x1)",
     main = "Histogram of Input Audio (x1)")

# Histogram of MEG signal (y)
hist(brainData$y,
     xlab = "MEG Signal (y)",
     main = "Histogram of MEG Signal (y)")

# Density plot of input audio (x1)
plot(density(brainData$x1),
     xlab = "Input Audio (x1)",
     main = "Density Plot of Input Audio (x1)")

# Density plot of MEG signal (y)
plot(density(brainData$y),
     xlab = "MEG Signal (y)",
     main = "Density Plot of MEG Signal (y)")

# Correlation between input audio (x1) and MEG signal (y)
audioBrainSignalCorrelation <- cor(brainData$x1, brainData$y)
print(paste("Correlation between x1 and y is:", audioBrainSignalCorrelation))

# Scatter plot of input audio (x1) vs. MEG signal (y)
plot(brainData$x1,
     brainData$y,
     xlab = "Input Audio (x1)",
     ylab = "MEG Signal (y)",
     main = "Scatter Plot of Input Audio (x1) vs. MEG Signal (y)")

# Boxplot of MEG signal (y) by audio category (x2)
boxplot(brainData$y ~ brainData$x2,
        xlab = "Audio Category (x2)",
        ylab = "MEG Signal (y)",
        main = "Boxplot of MEG Signal (y) by Audio Category (x2)")

# Filter data for neutral audio (x2 == 0)
neutralAudio <- brainData[brainData$x2 == 0, ]
head(neutralAudio)

# Time series plot of MEG signal (y) for neutral audio
plot(neutralAudio$time,
     neutralAudio$y,
     type = "l",
     xlab = "Time",
     ylab = "MEG Signal (y) - Neutral",
     main = "Time Series of MEG Signal (y) - Neutral Audio")

# Histogram of MEG signal (y) for neutral audio
hist(neutralAudio$y,
     xlab = "MEG Signal (y) - Neutral",
     main = "Histogram of MEG Signal (y) - Neutral Audio")

# Filter data for emotional audio (x2 == 1)
emotionalData <- brainData[brainData$x2 == 1, ]

# Time series plot of MEG signal (y) for emotional audio
plot(emotionalData$time,
     emotionalData$y,
     type = "l",
     xlab = "Time",
     ylab = "MEG Signal (y) - Emotional",
     main = "Time Series of MEG Signal (y) - Emotional Audio")

# Histogram of MEG signal (y) for emotional audio
hist(emotionalData$y,
     xlab = "MEG Signal (y) - Emotional",
     main = "Histogram of MEG Signal (y) - Emotional Audio")

# -------------------------Task2-------------------------
# Task 2.1 Estimating model parameters
parameterEstimation <- function(x1, x2, modelNumber) {
  if (modelNumber == 1) {
    X <- cbind(x1^3, x1^5, x2, 1)
  } else if (modelNumber == 2) {
    X <- cbind(x1, x2, 1)
  } else if (modelNumber == 3) {
    X <- cbind(x1, x1^2, x1^4, x2, 1)
  } else if (modelNumber == 4) {
    X <- cbind(x1, x1^2, x1^3, x1^5, x2, 1)
  } else if (modelNumber == 5) {
    X <- cbind(x1, x1^3, x1^4, x2, 1)
  } else {
    stop("Invalid model number")
  }
  return(X)
}

modelParameters <- list()
for (i in 1:5) {
  X <- parameterEstimation(brainData$x1, brainData$x2, i)
  y <- brainData$y
  
  parameters <- solve(t(X) %*% X) %*% t(X) %*% y
  modelParameters[[i]] <- parameters
  
  cat("Parameters for Model", i)
  print(parameters)
  cat("\n")
}

# Task 2.2 Residual Sum of Squares
RSSvalues <- numeric(5)
for (i in 1:5) {
  X <- parameterEstimation(brainData$x1, brainData$x2, i)
  yPredicted <- X %*% modelParameters[[i]]
  residualValues <- brainData$y - yPredicted
  RSS <- sum(residualValues^2)
  RSSvalues[i] <- RSS
  
  cat("Residual Sum of Squares (RSS) for Model", i, ":", RSS, "\n")
}

cat("RSS Values for all Models:", RSSvalues, "\n")

# Task 2.3: Compute Log-Likelihood

logLikelihoodValues <- numeric(5)
n <- length(brainData$y)

for (i in 1:5) {
  X <- parameterEstimation(brainData$x1, brainData$x2, i)
  yPredicted <- X %*% modelParameters[[i]]
  residualValues <- brainData$y - yPredicted
  RSS <- sum(residualValues^2)
  sigma2Estimation <- RSS / n
  logLikelihood <- -n/2 * log(2 * pi) - n/2 * log(sigma2Estimation) - 1/(2 * sigma2Estimation) * RSS
  logLikelihoodValues[i] <- logLikelihood
  
  cat("Log-Likelihood for Model", i, ":", logLikelihood, "\n")
}

# Task 2.4: Compute AIC and BIC

AICvalues <- numeric(5)
BICvalues <- numeric(5)
numberofdatapoints <- length(brainData$y)


kValues <- c(4, 3, 5, 6, 5)

for (i in 1:5) {
  log_likelihood <- logLikelihoodValues[i]
  k <- kValues[i]
  
  AIC <- 2 * k - 2 * log_likelihood
  BIC <- k * log(numberofdatapoints) - 2 * log_likelihood
  
  AICvalues[i] <- AIC
  BICvalues[i] <- BIC
  
  cat("Model", i, "- AIC:", AIC, ", BIC:", BIC, "\n")
}

cat("AIC Values for all Models:", AICvalues, "\n")
cat("BIC Values for all Models:", BICvalues, "\n")


# Task 2.5: Check the distribution of model prediction errors (residuals)

calculateResiduals <- function(x1, x2, y, model_params, model_number) {
  X <- parameterEstimation(x1, x2, model_number)
  yPredicted <- X %*% model_params
  residuals <- as.vector(as.numeric(y - yPredicted))  # Ensure residuals are a numeric vector
  return(residuals)
}

# Visualize residuals using histograms and Q-Q plots
par(mfrow = c(2, 3))

for (i in 1:5) {
  
  residuals <- calculateResiduals(brainData$x1, brainData$x2, brainData$y, modelParameters[[i]], i)
  
  residuals_simple <- as.vector(as.numeric(residuals))
  attributes(residuals_simple) <- NULL
  
  # Histogram of residuals
  try({
    hist(residuals_simple,
         main = paste("Model", i, "Residuals Histogram"),
         xlab = "Residuals",
         ylab = "Frequency",
         col = "lightblue",
         border = "black")
  }, silent = TRUE)
  
  # Q-Q plot of residuals
  try({
    qqnorm(residuals_simple,
           main = paste("Model", i, "Residuals Q-Q Plot"),
           xlab = "Theoretical Normal Quantiles",
           ylab = "Sample Quantiles")
    qqline(qqnorm(residuals_simple, plot.it = FALSE))
  }, silent = TRUE)
  
  try({
    shapiroTest <- shapiro.test(residuals_simple)
    cat("Model", i, "Shapiro-Wilk Test p-value:", shapiroTest$p.value, "\n\n")
  }, silent = TRUE)
}

# Task 2.6 Selecting the best model
par(mfrow = c(1, 1))

cat("Residual Sum of Squares (RSS):\n")
print(RSSvalues)

cat("\nLog-Likelihood Values:\n")
print(logLikelihoodValues)

cat("\nAIC Values:\n")
print(AICvalues)

cat("\nBIC Values:\n")
print(BICvalues)

cat("\nShapiro-Wilk Test p-values:\n")
for (i in 1:5) {
  try({
    shapiro_test <- shapiro.test(calculateResiduals(brainData$x1, brainData$x2, brainData$y, modelParameters[[i]], i))
    cat("Model", i, ":", shapiro_test$p.value, "\n")
  }, silent = TRUE)
}


summary(lm(brainData$y ~ poly(brainData$x1, 2) + brainData$x2)) # Detailed summary of the best fitting model (Model 3)

# Task 2.7
# Split the data into training and testing sets
set.seed(123)

n_total <- length(brainData$y)
n_train <- floor(0.7 * n_total)

trainIndices <- sample(1:n_total, n_train, replace = FALSE)

trainData <- brainData[trainIndices, ]
testData <- brainData[-trainIndices, ]

train_x1 <- trainData$x1
train_x2 <- trainData$x2
train_y <- trainData$y

test_x1 <- testData$x1
test_x2 <- testData$x2
test_y <- testData$y

# Estimate model parameters
train_X <- model.matrix(~ poly(train_x1, 2) + train_x2)

trainModelParams <- solve(t(train_X) %*% train_X) %*% t(train_X) %*% train_y

cat("Trained parameters for Model 3 (using training data):\n")
print(trainModelParams)

# Compute models prediction on testing data
test_X <- model.matrix(~ poly(test_x1, 2) + test_x2)
predicted_y <- test_X %*% trainModelParams

# Compute the 95% model prediction confidence intervals
trainModellm <- lm(train_y ~ poly(train_x1, 2) + train_x2)
residualSE <- summary(trainModellm)$sigma

alpha <- 0.05
n_test <- nrow(test_X)
predictionIntervals <- matrix(NA, nrow = n_test, ncol = 2)

for (i in 1:n_test) {
  x_i <- test_X[i, , drop = FALSE]
  predictionVariance <- residualSE^2 * (1 + x_i %*% solve(t(train_X) %*% train_X) %*% t(x_i))
  predictionSE <- sqrt(predictionVariance)
  criticalValue <- qt(1 - alpha / 2, df = nrow(train_X) - ncol(train_X)) # Degrees of freedom
  
  predictionIntervals[i, 1] <- predicted_y[i] - criticalValue * predictionSE
  predictionIntervals[i, 2] <- predicted_y[i] + criticalValue * predictionSE
}

# Plotting
plot(test_x1, test_y, pch = 18, col = ifelse(test_x2 == 0, "blue", "red"),
     xlab = "Input Audio Signal (x1)", ylab = "MEG Response (y)",
     main = "Model 3 Prediction with 95% Confidence Intervals (Testing Data)")

# Order the test data by x1 for better visualization of the prediction line
orderIndices <- order(test_x1)
sortedTest_x1 <- test_x1[orderIndices]
sortedPredicted_y <- predicted_y[orderIndices]
sortedLowerBound <- predictionIntervals[orderIndices, 1]
sortedUpperBound <- predictionIntervals[orderIndices, 2]

lines(sortedTest_x1, sortedPredicted_y, col = "green", lwd = 2)

# Add error bars for the confidence intervals
segments(sortedTest_x1, sortedLowerBound, sortedTest_x1, sortedUpperBound, col = "darkgrey", lty = "dashed")

legend("topleft", legend = c("Testing Data (Neutral Audio)", "Testing Data (Emotional Audio)", "Model 3 Prediction", "95% Prediction Interval"),
       col = c("blue", "red", "green", "darkgrey"), pch = c(16, 16, NA, NA), lty = c(NA, NA, 1, 2))

# Print the summary of the model trained on the training data
print(summary(trainModellm))

# Calculate residuals on the testing data
testResiduals <- predicted_y - test_y
mse_test <- mean(testResiduals^2)
rmse_test <- sqrt(mse_test)
mae_test <- mean(abs(testResiduals))

sst_test <- sum((test_y - mean(test_y))^2)
ssr_test <- sum(testResiduals^2)
rSquaredTest <- 1 - (ssr_test / sst_test)

# Print all metrics
cat("--- Model Performance on Testing Data ---\n")
cat("Mean Squared Error (MSE):", mse_test, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_test, "\n")
cat("Mean Absolute Error (MAE):", mae_test, "\n")
cat("R-squared:", rSquaredTest, "\n")
cat("-----------------------------------------\n")

# -------------------------Task3: Approximate Bayesian Computation (ABC)-------------------------
# taken 20% of coefficients poly(train_x1, 2)1 and poly(train_x1, 2)2 in summary of trainModellm
priorParam1 <- c(80.25, 120.38) 
priorParam2 <- c(62.41, 93.61)  

nIterations <- 10000
tolerance <- 5.0

acceptedParam1 <- numeric()
acceptedParam2 <- numeric()

#values taken from task 2.7
fixedIntercept <- 16.7047
fixedX21 <- 4.5980
residualSd <- 3.244

library(splines)
library(ggplot2)

for (i in 1:nIterations) {

  sampledParam1 <- runif(1, min = priorParam1[1], max = priorParam1[2])
  sampledParam2 <- runif(1, min = priorParam2[1], max = priorParam2[2])
  
  simulatedY <- fixedIntercept +
    sampledParam1 * poly(train_x1, 2)[, 1] +
    sampledParam2 * poly(train_x1, 2)[, 2] +
    fixedX21 * as.numeric(as.character(train_x2)) + # Corrected line: Convert factor to numeric
    rnorm(length(train_y), mean = 0, sd = residualSd)
  
  rmseSimulated <- sqrt(mean((simulatedY - train_y)^2))
  
  if (rmseSimulated < tolerance) {
    acceptedParam1 <- c(acceptedParam1, sampledParam1)
    acceptedParam2 <- c(acceptedParam2, sampledParam2)
  }
}

cat("Number of accepted samples for ABC:", length(acceptedParam1), "\n")

posteriorDf <- data.frame(param1 = acceptedParam1, param2 = acceptedParam2)

jointPosteriorPlot <- ggplot(posteriorDf, aes(x = param1, y = param2)) +
  geom_point(alpha = 0.5) +
  xlab("Coefficient of poly(train_x1, 2)1") +
  ylab("Coefficient of poly(train_x1, 2)2") +
  ggtitle("Joint Posterior Distribution (ABC)")
print(jointPosteriorPlot)

marginalParam1Plot <- ggplot(posteriorDf, aes(x = param1)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(alpha = 0.2, fill = "lightcoral") +
  xlab("Coefficient of poly(train_x1, 2)1") +
  ylab("Density") +
  ggtitle("Marginal Posterior Distribution of poly(train_x1, 2)1 (ABC)")
print(marginalParam1Plot)

marginalParam2Plot <- ggplot(posteriorDf, aes(x = param2)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(alpha = 0.2, fill = "lightcoral") +
  xlab("Coefficient of poly(train_x1, 2)2") +
  ylab("Density") +
  ggtitle("Marginal Posterior Distribution of poly(train_x1, 2)2 (ABC)")
print(marginalParam2Plot)

