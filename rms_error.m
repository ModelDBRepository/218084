function [rmse] = rms_error(y,f)
%computes the root mean squared error, betweena actual data 'y' and model
%data 'f'
rmse = sqrt(mean((y(:) - f(:)).^2));
