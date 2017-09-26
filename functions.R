############ functions ################

binarize <- function(datax){
  for (i in 1:dim(datax)[1]){
    for(k in 1:dim(datax)[2]){
      if(datax[i,k]/255 > 0.5){
        datax[i,k] <- 1
      }
      else if (datax[i,k]/255 < 0.5){
        datax[i,k] <- 0
      }
    }
  }
  datax
}



get_thetamap <- function(datax, classvector){
  pixelcount1 <- matrix(0, nrow = 784, ncol = 10)
  pixelcount0 <- matrix(0, nrow = 784, ncol = 10)
  for (d in 1:784){
    for (c in 1:10){
      pixelcount1[d,c] <- sum(datax[classvector ==c,d] == 1)
      pixelcount0[d,c] <- sum(datax[classvector ==c,d] == 0)
      
    }
  }
  thetamap <- (pixelcount1 + 1)/(pixelcount0 + pixelcount1 + 2)
  thetamap
}



plot_image <- function(image){
  grays = rgb(red = 0:255/255, blue = 0:255/255, green = 0:255/255)
  imagematrix <- matrix(image, nrow = 28, ncol = 28)
  heatmap(t(imagematrix),Rowv=NA,Colv=NA,col=grays, scale = "none", revC = T, xlab = '', ylab = '', margins = c(0,0), keep.dendro = F)
}
help(heatmap)


pixelcount1 <- matrix(0, nrow = 784, ncol = 10)
pixelcount0 <- matrix(0, nrow = 784, ncol = 10)
for (d in 1:784){
  for (c in 1:10){
    pixelcount1[d,c] <- sum(trainx[trainy ==c,d] == 1)
    pixelcount0[d,c] <- sum(trainx[trainy ==c,d] == 0)
  }
}

################################################################################bayes##
predict_class <- function(thetamap, testimage){
  tmp <- rep(0, 784)
  x <- rep(0, 10)
  for (c in 1:10){
    for (d in 1:784){
      tmp[d] <- log(1/10) + testimage[d]*log(thetamap[d, c]) + (1- testimage[d])*log(1 - (thetamap[d, c]))
    }
    x[c] <- sum(tmp)
  }
  relative_log_prob <- rep(0, 10)
  for(c in 1:10){
    relative_log_prob[c] <- x[c] - logsumexp(x[-c]) 
  }
  pred <- which(relative_log_prob == max(relative_log_prob)) 
  c(relative_log_prob,pred)
}

max(trainy)


logsumexp <- function(a){
  m = max(a)
  log(sum(exp(a-m))) + m
}


test_model <- function(testdata, labels, thetamap){
  predictions <- matrix(0, nrow = 11, ncol = length(labels))
  correct <- rep(0, length(labels))
  avg_likelihood <- rep(0, length(labels))
  for(i in 1:length(labels)){
    predictions[,i] <- predict_class(thetamap, testdata[i,])
    correct[i] <- predictions[11,i]==labels[i]
    avg_likelihood[i] <- sum(predictions[-11, i])/10
  }
  accuracy <- sum(correct)/length(labels)
  c(accuracy, avg_likelihood)
}
###function returns vector with index[1] = accuracy and rest is mean pred log likelihood

############LOGISTIC ################################################

predict_class_logistic <- function(w, image){
  logclassprob <- rep(0, 10)
  for(i in 1:10){
    logclassprob[i] <- log(exp(w[,i]%*%image)/sum(exp((t(w[,-i])%*%image))))
  }
  guess <- which(logclassprob == max(logclassprob))
  c(logclassprob, guess)
}

grad <- function(weights, imagelabel, image){
  probc <- rep(0, 10)
  gradient <- matrix(0, nrow = 784, ncol = 10)
  for (k in 1:10){
    probc[k] <- exp((weights[,k]%*%image))/exp(logsumexp((t(weights)%*%image)))
  }
  for(i in 1:10){
    gradient[,i] <- imagelabel[i]*image - image*probc[i]
  }
  gradient
}

rm(image)
str(w + g1[[1]]*stepsize)

t(w + g1[[1]]*stepsize)%*%image



logistic_likelihood <- function(w, image){
  loglike <- rep(0, 10)
  for(c in 1:10){
    loglike[c] <- ((w[,c])%*%image)/logsumexp(t(w)%*%image)
  }
  loglike
}



predict_class_logistic(w, image)
plot_image(image)



grad2 <- function(weights, imagelabels, imagedata){
  probc <- rep(0, 10)
  gradient <- matrix(0, nrow = 784, ncol = 10)
  for (k in 1:10){
    probc[k] <- (weights[,k]%*%t(imagedata))/sum(exp((t(weights)%*%t(imagedata))))
  }
  for(i in 1:10){
    gradient[,i] <- imagelabels[i,]*sum(imagedata - image*probc[i])
  }
  gradient
}






predict_class_logistic <- function(w, image){
  logclassprob <- rep(0, 10)
  logclassprob <- logistic_likelihood(w, image)
  guess <- which(logclassprob == max(logclassprob))
  c(logclassprob, guess)
}

#######################