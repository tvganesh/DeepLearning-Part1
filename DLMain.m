source("DL-functions.m")

cancer=csvread("cancer.csv");
[X_train,y_train,X_test,y_test] = trainTestSplit(cancer,75);

w=zeros(size(X_train)(1,2),1);
b=0;
X_train1=normalize(X_train);
X_train2=X_train1';
y_train1=y_train';
[w1,b1,dw,db,losses,idx]=gradientDescent(w, b, X_train2, y_train1, numIerations=3000, learningRate=0.75);

# Normalize X_test
X_test1=normalize(X_test);
#Transpose X_train so that we have a matrix as (features, numSamples)
X_test2=X_test1';
y_test1=y_test';

# Use the values of the weights generated from Gradient Descent
yPredictionTest = predict(w1, b1, X_test2);

yPredictionTrain = predict(w1, b1, X_train2);

trainAccuracy=100-mean(abs(yPredictionTrain - y_train1))*100
testAccuracy=100- mean(abs(yPredictionTest - y_test1))*100

graphics_toolkit('gnuplot')
plot(idx,losses);
title ('Gradient descent- Cost vs No of iterations');
xlabel ("No of iterations");
ylabel ("Cost");
print ("lr.jpg") 