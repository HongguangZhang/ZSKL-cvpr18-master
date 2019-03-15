% rbf.m This function returns the function value, partial derivatives
% and Hessian of the (general dimension) rosenbrock function, given by:

function [f] = rbf(W, x, y, derta)
   % x: 2048x1; 
   % W: 2048x85;
   % y: 85x1
   f = exp(-sum((W'*x - y).^2)/(2*derta^2));
end



