log_likelihood <- function(thetamap, image, c){
  tmp <- rep(0, 784)
  for(d in 1:784){
    tmp[d] <- image[d]*log(thetamap[d,c]) + (1-image[d])*log(1-thetamap[d,c]) + log(1/10)
    }
  sum(tmp)
}


predict_bayes <- function(thetamap, image){
  likelihoods <- rep(0, 10)
  for(c in 1:10){
    likelihoods[c] <- log_likelihood(thetamap, image, c)
  }
  marginal <- logsumexp(likelihoods)
  predictivelikelihood <- rep(0, 10)
  for(c in 1:10){
    predictivelikelihood[c] <- likelihoods[c] - marginal
    
  }  
  prediction <- which(predictivelikelihood == max(predictivelikelihood))
  average <- predictivelikelihood[1]
  data.frame(prediction, average)
}



test_bayes <- function(thetamap, testdata, testlabels){
  N <- length(testlabels)
  guesses <- rep(0, N)
  avgpredlikelihood <- rep(0, N)
  for(i in 1:N){
    guesses[i] <- predict_bayes(thetamap, testdata[i,])$prediction == testlabels[i]
    avgpredlikelihood[i] <- predict_bayes(thetamap, testdata[i,])$average
  }
  accuracy <<- sum(guesses)/N
  meanpredictiveloglikelihood <<- avgpredlikelihood
}



pcgivenxtop <- function(thetamap, tophalf){
  probs <- rep(0, 10)
  for(i in 1:10){
    probs[i] <- log_likelihood(thetamap, c(tophalf, rep(0, 392)), i)
    
  }
  marginal <- logsumexp(probs)
  predictivelikelihood <- rep(0, 10)
  for(c in 1:10){
    predictivelikelihood[c] <- probs[c] - marginal
    
  }
  exp(predictivelikelihood)
}





