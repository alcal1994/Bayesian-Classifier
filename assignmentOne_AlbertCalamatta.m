
%Histogram for classOne data
histogram(classOne,100);
hold on
%Histogram for classTwo data
histogram(classTwo,100);

%Add title, x/y axis labels, and legend
title('Class One and Two data');
xlabel('Sample Value');
ylabel('Number of Samples');
legend('Class One', 'Class Two');

%Find probability of class one

%Find length of class one
numClassOneVariables = length(classOne);

%Find length of class two
numClassTwoVariables = length(classTwo);

%create variable to hold total number of class one and two variables
totalNumClassVariables = numClassOneVariables + numClassTwoVariables;

%probability of classOne
probClassOne = numClassOneVariables/totalNumClassVariables;

%probability of classTwo
probClassTwo = numClassTwoVariables/totalNumClassVariables;

%split classOne sample data into training and testing partitions

%randomize indices between 1 and 10000
rng(1);
indices = randsample(10000,10000);

%set indices for training set after they've been randomized
trainingSet = indices(1:6000);
testingSet = indices(6001:10000);

%Find mean of training data
meanClassOne = mean(classOne(trainingSet));
meanClassTwo = mean(classTwo(trainingSet));

%test print for mean of training sets
%disp(meanClassOne);
%disp(meanClassTwo);

%find standard deviation of training data 
standardDeviationClassOne = std(classOne(trainingSet));
standardDeviationClassTwo = std(classTwo(trainingSet));

%test print for standard deviation of classOne training set
%disp(standardDeviationClassOne);
%disp(standardDeviationClassTwo);

%classify each of the samples in tesing partition using bayesian classifier
% m is the mean, S is the variance or covariance matrix, x is the feature
% vector or singular value (taken from the Pattern Recognition for
% consistency)

for i = 1:4000
    y(i) = computeGaussianDensity(meanClassOne,standardDeviationClassOne,testingSet(i));
    z(i) = computeGaussianDensity(meanClassTwo,standardDeviationClassTwo,testingSet(i));
    
    if(y(i) > z(i))
        predictedClass(i) = 1;
    else
        predictedClass(i) = 2;
    end
    
end

correctClassOnePredictions = sum(predictedClass == 1);
incorrectClassOnePredictions = sum(predictedClass == 2);

for i = 1:4000
    z(i) = computeGaussianDensity(meanClassTwo,standardDeviationClassTwo,testingSet(i));
    y(i) = computeGaussianDensity(meanClassOne,standardDeviationClassOne,testingSet(i));
    
    if(z(i) > y(i))
        predictedClass(i) = 1;
    else
        predictedClass(i) = 2;
    end
end

correctClassTwoPredictions = sum(predictedClass == 2);
incorrectClassTwoPredictions = sum(predictedClass == 1);

totalPredictionAccuracy = (correctClassOnePredictions + correctClassTwoPredictions) / 8000;
disp(totalPredictionAccuracy);

function z=computeGaussianDensity(m,S,x)
 
[l,q] = size(m);  %l dimensionality
z = (1/((2*pi)^l/2*det(S)^0.5)) * exp(-0.5 * (x-m)' * inv(S) * (x-m));
end











