function [f,f_p] = f_b(x,type)
%function m = f_b(x)
%
%This function calcualtes an activation function to a layer of neurons, as
%well as it's derivative
%
% x -
% This is a column vector of neurons from a particular layer.

switch type
    case 'lin'
        f = x;
        f_p = ones(size(x));
    case 'tanh'
        f = tanh(x);
        f_p = ones(size(x)) - f.^2;
    case 'logsig'
        f = 1./ (ones(size(x)) + exp(-x));
        f_p = f .* (ones(size(x)) - f) ;
    case 'reclin'
        f = max(x,0);
        f_p = sign(f);
    otherwise
        f = x;
        f_p = ones(size(x));
end
end
