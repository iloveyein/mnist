##############################naive bayes train data#####################
trainx <- train$x[1:10000,]
trainy <- train$y[1:10000]
trainy <- trainy + 1

newdatax <- binarize(trainx)
thetamap <- get_thetamap(newdatax, trainy)

imagematrix <- list()
grays = rgb(red = 0:255/255, blue = 0:255/255, green = 0:255/255)
for(i in 1:10){
  imagematrix[[i]] <- matrix(thetamap[,i], nrow = 28, ncol = 28)
  heatmap(t(imagematrix[[i]]),Rowv=NA,Colv=NA,col=grays, scale = "none", revC = T)
}
###############naive bayes predict class for each image##################

testx <- binarize(test$x)
testx <- t(testx)
test$y <- test$y + 1

test_bayes(thetamap, newdatax, trainy)
trainacc <- accuracy
trainlikelihoods <- meanpredictiveloglikelihood

sum(meanpredictiveloglikelihood)/1000000

test_bayes(thetamap, testx, test$y)
testacc <- accuracy
testlikelihoods <- meanpredictiveloglikelihood

###sample and plot 10 binary images from marginal distribution #################
set.seed(980604)
classsample <- sample(1:10,10, replace = T)


imagesample <- list()
image <- list()
for (i in 1:10){
  imagesample[[i]] <- rbinom(784, 1, thetamap[,classsample[i]])
  plot_image(imagesample[[i]])
}

### plot top half of image concatenated with marginal distribution over each pixel in bottom half ###
small <- newdatax[1:20,]
smallclass <- trainy[1:20]
top <- small[,1:392]



fimage <- matrix(0, ncol = 784, nrow = 20)
for(i in 1:20){
  fimage[i,] <- c(top[i,],thetamap[393:784,]%*%pcgivenxtop(thetamap, top[i,]))
  plot_image(fimage[i,])
}

###########TRAINING LOGISTIC REGRESSION #################

trainlabels <- matrix(0, nrow = 10, ncol = 10000)
for(i in 1:10000){
  trainlabels[,i] <- oneofk[, trainy[i]]
}

w <-  matrix(0, nrow = 784, ncol = 10)
stepsize <- 0.001
g1 <- rep(list(diag(0, nrow = 784, ncol = 10)), 1000)
batch <- rep(0, 1000)

for (i in 1:1000){
  batch[i] <- round(runif(1, min = 257, max = 10000))
}  

for(d in 1:1000){
  for(n in (batch[d] - 256):batch[d]){
    g <- grad(weights = w, trainlabels[,n], newdatax[n,])
    g1[[d]] <- g1[[d]] + g
  }
  w <- w + g1[[d]]*stepsize
}


trainresult <- matrix(0, nrow = 10000, ncol = 11)
for(n in 1:10000){
  trainresult[n,] <- predict_class_logistic(w, newdatax[n,])
}

sum(trainresult[,11] == trainy[1:10000])/10000

mean(result[,5])


logistictrainacc <- sum(trainresult[,11] == trainy[1:10000])/10000
train_avgpred_likelihood <- apply(trainresult[,1:10],1, FUN = max)
mean(log(train_avgpred_loglikelihood))

w[,1]
for(i in 1:10){
  plot_image(w[,i])
}

################# TESTING LOGISTIC REGRESSION ##################

testx <- binarize(test$x)
testx <- t(testx)
test$y <- test$y + 1 #skip above three steps if you already did for bayes important!


testlabels <- matrix(0, nrow = 10 , ncol = 10000)
for (i in 1:10000){
  testlabels[,i] <- oneofk[,test$y[i]]
}

testresult <- matrix(0, nrow = 10000, ncol = 11)
for(n in 1:10000){
  testresult[n,] <- predict_class_logistic(w, testx[,n])
}

logistictestacc <- sum(testresult[,11] == test$y)/10000
test_avgpred_loglikelihood <- apply(result[,1:10],1, FUN = max)
mean(log(test_avgpred_loglikelihood))



