%Find length of class one
numClassOneVariables = length(classOne);

%Find length of class two
numClassTwoVariables = length(classTwo);

%Display the length of class One
disp(numClassOneVariables);

%Display the length of class Two
disp(numClassTwoVariables);

%create total number of class data variable
total = numClassOneVariables + numClassTwoVariables;

%Find probability of classOne
probClassOne = numClassOneVariables/total;

%display class one probability
disp(probClassOne);

%Find probability of classTwo
probClassTwo = numClassTwoVariables/total;

%display class two probability
disp(probClassTwo);

%randomize indices between 1 and 10000
indices = randperm(10000);

%set indices for training set after they've been randomized
trainingSet = classOne(indices(1:6000),:);
trainingSetTwo = classTwo(indices(1:6000),:);

%set indices for testing set after they've been randomized
testingSet = classOne(indices(6001:10000),:);
testingSetTwo = classTwo(indices(6001:10000),:);

%transpose testing sets
testingSetTrans = transpose(testingSet);
testingSetTransTwo = transpose(testingSetTwo);

%Find mean of classes
meanClassOne = mean(trainingSet);
meanClassTwo = mean(trainingSetTwo);

%Display mean of classes
disp(meanClassOne);
disp(meanClassTwo);

%Covariance matrix
covClassOne = cov(trainingSet);
covClassTwo = cov(trainingSetTwo);

%display covariance matrix of classes
disp(covClassOne);
disp(covClassTwo);


for i=1:length(testingSet)
    z = computeGaussianDensityMultivariate(meanClassOne,covClassOne,testingSetTrans(:,1));
    disp(z);
end

for i=1:length(testingSet)
    z = computeGaussianDensityMultivariate(meanClassTwo,covClassTwo,testingSetTransTwo(:,1));
    disp(z);
end

function z=computeGaussianDensityMultivariate(m,S,x)
 
[l,q] = size(m);  %l dimensionality
z = (1/((2*pi)^l/2*det(S)^0.5)) * exp(-0.5 * (x-m)' * inv(S) * (x-m));

end



