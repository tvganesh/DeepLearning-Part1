1;
# Define sigmoid function
function a = sigmoid(z)
  a = 1 ./ (1+ exp(-z));
end
# Compute the loss
function loss=computeLoss(numtraining,Y,A)
  loss = -1/numtraining * sum((Y .* log(A)) + (1-Y) .* log(1-A));
end

# Perform forward propagation
function [loss,dw,db,dZ] = forwardPropagation(w,b,X,Y)
  % Compute Z
  Z = w' * X + b;
  numtraining = size(X)(1,2);
  # Compute sigmoid
  A = sigmoid(Z);
  
  #Compute loss. Note this is element wise product
  loss =computeLoss(numtraining,Y,A);
  # Compute the gradients dZ, dw and db
   dZ = A-Y;
   dw = 1/numtraining* X * dZ';
   db =1/numtraining*sum(dZ);
    
end

# Compute Gradient Descent
function [w,b,dw,db,losses,index]=gradientDescent(w, b, X, Y, numIerations, learningRate)
  #Initialize losses and idx
  losses=[];
  index=[];
  # Loop through the number of iterations
  for i=1:numIerations,
     [loss,dw,db,dZ] = forwardPropagation(w,b,X,Y);
     # Perform Gradient descent
     w = w - learningRate*dw;
     b = b - learningRate*db;
     if(mod(i,100) ==0)
        # Append index and loss
        index = [index i];
        losses = [losses loss];
     endif
     
  end
end

# Determine the predicted value for dataset
function yPredicted = predict(w,b,X)
   m = size(X)(1,2);
   yPredicted=zeros(1,m);
   # Compute Z
   Z = w' * X + b;
   # Compute sigmoid
   A = sigmoid(Z);
   for i=1:size(X)(1,2),
      # Set predicted as 1 if A > 0,5
      if(A(1,i) >= 0.5)
        yPredicted(1,i)=1;
       else
        yPredicted(1,i)=0;
       endif
    end
end

# Normalize by dividing each value by the sum of squares
function normalized = normalize(x)
    # Compute Frobenius norm. Square the elements, sum rows and then find square root
    a = sqrt(sum(x .^ 2,2));
    # Perform element wise division
    normalized = x ./ a;
end

# Split into train and test sets
function [X_train,y_train,X_test,y_test] = trainTestSplit(dataset,trainPercent)   
     # Create a random index
     ix = randperm(length(dataset));
     # Split into training
     trainSize = floor(trainPercent/100 * length(dataset));
     train=dataset(ix(1:trainSize),:);
     # And test
     test=dataset(ix(trainSize+1:length(dataset)),:);
     X_train = train(:,1:30);
     y_train = train(:,31);
     X_test = test(:,1:30);
     y_test = test(:,31);
end