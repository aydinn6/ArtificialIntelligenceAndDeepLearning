% Clear workspace, close all figures, and clear the command window
clear all, close all, clc;

% Define the path to the dataset (Digit Dataset from MATLAB's neural network toolbox)
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos','nndatasets','DigitDataset');

% Create an imageDatastore object that holds the dataset of images
% 'IncludeSubfolders' means it will include subfolders for each digit label
% 'LabelSource' is set to 'foldernames' to assign labels based on the folder names
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource','foldernames');

% Display 20 random images from the dataset to visualize some of the digits
figure;
perm = randperm(10000,20); % Randomly permute and select 20 images
for i = 1:20
    subplot(4,5,i); % Create a 4x5 grid of subplots
    imshow(imds.Files{perm(i)}); % Display the image at the selected index
end

% Count the number of images in each label (digit class)
labelCount = countEachLabel(imds);

% Read the first image from the image datastore to check its size
img = readimage(imds,1);
size(img)

% Define the number of training images for each class (750 per label)
numTrainFiles = 750;

% Split the dataset into training and validation sets
% 'randomize' ensures that the split is random
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');

% Define the layers for the convolutional neural network (CNN)
layers = [
    imageInputLayer([28 28 1]) % Input layer with 28x28 grayscale images
    convolution2dLayer(3,8,'Padding', 'same') % First convolution layer with 8 filters
    batchNormalizationLayer % Batch normalization to improve training stability
    reluLayer % ReLU activation function
    
    maxPooling2dLayer(2,'Stride',2) % Max pooling layer with 2x2 window
    
    convolution2dLayer(3,16,'Padding','same') % Second convolution layer with 16 filters
    batchNormalizationLayer % Batch normalization layer
    reluLayer % ReLU activation function
    
    maxPooling2dLayer(2,'Stride',2) % Max pooling layer with 2x2 window
    
    convolution2dLayer(3,32,'Padding','same') % Third convolution layer with 32 filters
    batchNormalizationLayer % Batch normalization layer
    reluLayer % ReLU activation function
    
    fullyConnectedLayer(10) % Fully connected layer with 10 units (one for each digit class)
    softmaxLayer % Softmax activation to output probabilities
    classificationLayer]; % Final classification layer to compute loss

% Set the options for training the neural network
options = trainingOptions('sgdm', ... % Stochastic Gradient Descent with Momentum
    'InitialLearnRate',0.01, ... % Set the initial learning rate
    'MaxEpochs',4, ... % Set the maximum number of training epochs
    'Shuffle','every-epoch', ... % Shuffle the data after each epoch
    'ValidationData',imdsValidation, ... % Set validation data for evaluating performance
    'ValidationFrequency',30, ... % Validate the model every 30 iterations
    'Verbose', false, ... % Do not display training progress in the command window
    'Plots', 'training-progress'); % Display a training progress plot

% Train the neural network using the training data and defined layers
net = trainNetwork(imdsTrain, layers, options);

% Classify the images in the validation dataset
YPred = classify(net, imdsValidation);

% Get the actual labels for the validation data
YValidation = imdsValidation.Labels;

% Calculate the accuracy by comparing predicted labels with the true labels
accuracy = sum(YPred == YValidation) / numel(YValidation);
