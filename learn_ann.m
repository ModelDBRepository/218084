function [w,b] = learn_ann(in,out,w,b,params)
%function [w,b] = learn_ann(in,out,w,b,params)
% w,b - these are the weights and biases
% in - input data, in an array of size (input data dimension x number of data samples) 
% out - output data, in an array of size (output data dimension x number of data samples)
% params - a structure containing parameters
l_rate = params.l_rate;
d_rate = params.d_rate;
type = params.type;
n_layers = params.n_layers;
a = n_layers-1;

iterations = length(in);
for its = 1:iterations
    x = cell(n_layers,1);
    d = cell(n_layers,1);
    f_p = cell(n_layers,1);
    f_n = cell(n_layers,1);
    grad_b = cell(size(b));
    grad_w = cell(size(w));
    
    %organise date into cells arrays
    x{1} = in(:,its);
    x_out = out(:,its); 
    %make a prediciton
    for ii = 2:n_layers
        [f_n{ii-1}, f_p{ii-1}] = f_b( x{ii-1}, type) ;
        x{ii} = w{ii-1} * ( f_n{ii-1} ) +  b{ii-1} ;
    end
    %backprop errors
    d{n_layers} = x_out - x{n_layers} ;
    for ii=n_layers-1:-1:2
        d{ii} = (w{ii}' * d{ii+1}) .* f_p{ii} ;
    end
    %find gradients
    for ii = 1:a
        grad_b{ii} = d{ii+1};
        grad_w{ii} = d{ii+1} * f_n{ii}' - d_rate*w{ii} ;
    end
    %update weights
    for ii = 1:a
        w{ii} = w{ii} + l_rate * grad_w{ii}   ;
        b{ii} = b{ii} + l_rate * grad_b{ii}   ;
    end
end
end