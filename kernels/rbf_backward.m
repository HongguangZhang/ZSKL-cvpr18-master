% rbf.m This function returns the function value, partial derivatives
% and Hessian of the (general dimension) rosenbrock function, given by:

function [df] = rbf_backward(W, x, y, derta)
   % x: 2048x1; 
   % W: 2048x85;
   % y: 85x1
   f = rbf(W, x, y, derta);
   df = -(1/derta^2).*f.*x*(W'*x - y)';
end



