source("RFunctions-1.R")
# Define the sigmoid function
sigmoid <- function(z){
    a <- 1/(1+ exp(-z))
    a
}

# Compute the loss
computeLoss <- function(numTraining,Y,A){
    loss <- -1/numTraining* sum(Y*log(A) + (1-Y)*log(1-A))
    return(loss)
}

# Compute forward propagation
forwardPropagation <- function(w,b,X,Y){
    # Compute Z
    Z <- t(w) %*% X +b
    #Set the number of samples
    numTraining <- ncol(X)
    # Compute the activation function
    A=sigmoid(Z) 
    
    #Compute the loss
    loss <- computeLoss(numTraining,Y,A)
    
    # Compute the gradients dZ, dw and db
    dZ<-A-Y
    dw<-1/numTraining * X %*% t(dZ)
    db<-1/numTraining*sum(dZ)
    
    fwdProp <- list("loss" = loss, "dw" = dw, "db" = db)
    return(fwdProp)
}

# Perform one cycle of Gradient descent
gradientDescent <- function(w, b, X, Y, numIerations, learningRate){
    losses <- NULL
    idx <- NULL
    # Loop through the number of iterations
    for(i in 1:numIerations){
        fwdProp <-forwardPropagation(w,b,X,Y)
        #Get the derivatives
        dw <- fwdProp$dw
        db <- fwdProp$db
        #Perform gradient descent
        w = w-learningRate*dw
        b = b-learningRate*db
        l <- fwdProp$loss
        # Stoe the loss
        if(i %% 100 == 0){
            idx <- c(idx,i)
            losses <- c(losses,l)  
        }
    }
    
    # Return the weights and losses
    gradDescnt <- list("w"=w,"b"=b,"dw"=dw,"db"=db,"losses"=losses,"idx"=idx)
    
    
    return(gradDescnt)
}

# COmpute the predicted value for input
predict <- function(w,b,X){
    m=dim(X)[2]
    # Create a ector of 0's
    yPredicted=matrix(rep(0,m),nrow=1,ncol=m)
    Z <- t(w) %*% X +b
    # Compute sigmoid
    A=sigmoid(Z)
    for(i in 1:dim(A)[2]){
        # If A > 0.5 set value as 1
        if(A[1,i] > 0.5)
            yPredicted[1,i]=1
        else
            # Else set as 0
            yPredicted[1,i]=0
    }
    
    return(yPredicted)
}

# Normalize the matrix
normalize <- function(x){
    #Create the norm of the matrix.Perform the Frobenius norm of the matrix 
    n<-as.matrix(sqrt(rowSums(x^2)))
    #Sweep by rows by norm. Note '1' in the function which performing on every row
    normalized<-sweep(x, 1, n, FUN="/")
    return(normalized)
}