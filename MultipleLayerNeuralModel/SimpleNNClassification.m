% Clear workspace and close figures
clear all; 
close all;

% Inputs
G = [1; 2; -1]; % Inputs G1, G2, G3
Wg = [1 1 1];   % Weights Wg-1, Wg-2, Wg-3

% Weights for first hidden layer
Wg_a1 = [3.5 2 2; 
                    1 -1 2; 
                    2.5 2 1; 
                    1 0 1]; 

% Calculate net inputs and outputs for first hidden layer
net1 = Wg_a1 * G; 
output1 = net1 >= 0; % Output = 1 if net >= 0, else 0

% Weights for second hidden layer
Wa1_a2 = [1 -1 1 1; 
                     2 1 2 0; 
                    -1.5 1 0 1]; 

% Calculate net inputs and outputs for second hidden layer
net2 = Wa1_a2 * output1; 
output2 = net2 >= 0; % Output = 1 if net >= 0, else 0

% Weights between second hidden layer and output layer
Wa2_output = [-3 3 -1]; 

% Calculate net input for output layer
net_output = Wa2_output * output2; 

% Activation function for output layer
output_final = net_output >= 1; % Final output = 1 if net >= 1, else -1
output_final = 2 * output_final - 1; % Convert boolean to -1/1

% Display results
disp('Net input for first hidden layer:');
disp(net1);
disp('Output of first hidden layer:');
disp(output1);
disp('Net input for second hidden layer:');
disp(net2);
disp('Output of second hidden layer:');
disp(output2);
disp('Net input for output layer:');
disp(net_output);
disp('Final Output:');
disp(output_final);
