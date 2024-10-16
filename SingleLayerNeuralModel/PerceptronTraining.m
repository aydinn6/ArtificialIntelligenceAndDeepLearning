clc, clear all, close all;
% Initial values
w = [0.5 0.2];      % Initial weights
Q = 0.4;                 % Threshold value
a = 0.9;                  % Learning rate
x1 = [1 0];             % Input data (example: Passed (GO))
x2 = [0 1];             % Input data (example: Failed (NOGO))
c1 = 1;                    % Expected output for Passed
c2 = -1;                   % Expected output for Failed

% Start of the training loop
for i = 1:100
    % Calculate net value and output for input x1
    net1 = (w * x1') + Q;  % Net value
    g1 = 1 * (net1 >= 0) + -1 * (net1 < 0); % Actual output
    
    % Calculate error and update weights and threshold for x1
    e1 = c1 - g1;                    % Calculate error
    if e1 ~= 0
        w = w + a * e1 * x1;  % Update weights
        Q = Q + a * e1;            % Update threshold value
    end
    
    % Calculate net value and output for input x2
    net2 = (w * x2') + Q;
    g2 = 1 * (net2 > 0) + -1 * (net2 <= 0);         % Actual output
    
    % Calculate error and update weights and threshold for x2
    e2 = c2 - g2;
    if e2 ~= 0
        w = w + a * e2 * x2;
        Q = Q + a * e2;
    end
    
    % If errors are zero, terminate the training loop
    if e1 == 0 && e2 == 0
        break;
    end
end

% Print results to the screen
fprintf('Training completed in %d steps.\n', i);
disp('New weights:');
disp(w);
disp('New threshold value:');
disp(Q);
