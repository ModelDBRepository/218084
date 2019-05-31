function [x,e,its] = infer_pc(x,w,b,params)
%function [x,e,its] = infer_pc(x,w,b,params)
% w,b - these are the weights and biases
% x - Variable nodes: First cell is input layer. Last cell is output layer
% e - Error nodes: First cell empty. Last cell is output layer
% params - a structure containing parameters
it_max = params.it_max;
n_layers = params.n_layers;
type = params.type;
beta = params.beta;
e = cell(n_layers,1);
f_n = cell(n_layers,1);
f_p = cell(n_layers,1);
var = params.var;

%calculate initial errors
for ii=2:n_layers
    [f_n{ii-1},f_p{ii-1}] = f_b( x{ii-1}, type) ;
    e{ii} = (x{ii} - w{ii-1} * ( f_n{ii-1} ) - b{ii-1})/var(ii) ;
end

for i = 1:it_max
    %update varaible nodes
    for ii=2:n_layers-1
        g = ( w{ii}' *  e{ii+1} ) .* f_p{ii} ;
        x{ii} = x{ii} + beta * ( - e{ii} + g );
    end
    %calculate errors
    for ii=2:n_layers
        [f_n{ii-1},f_p{ii-1}] = f_b( x{ii-1}, type) ;
        e{ii} = (x{ii} - w{ii-1} * ( f_n{ii-1} ) - b{ii-1})/var(ii) ;
    end     
end
its=i;
end