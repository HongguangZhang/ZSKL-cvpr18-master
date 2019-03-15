function [f, df, pred] = rbf_ort_loss(x, W, att, y, sigma, cls, phase, beta)
f = [];
N = numel(cls);
%pred = zeros(N,1,'single');
for i=1:N
    rbf_1(i) = rbf(W, x, att(:,cls(i)), sigma);
    rbf_2(i) = rbf(W', att(:, cls(i)), x, sigma);
    if cls(i) == y
        f(i) = (rbf_1(i) - 1)^2 + (rbf_2(i) - 1)^2;
    else
        f(i) = (beta/(N-1))*(rbf_1(i)^2 + rbf_2(i)^2);
    end
end

pred = (rbf_1-1).^2 + (rbf_2-1).^2;
f = sum(f);


df = zeros([size(W), N], 'single');
if strcmp(phase, 'train')
for i=1:N
    rbf_b1 = rbf_backward(W, x, att(:,cls(i)), sigma);
    rbf_b2 = rbf_backward(W', att(:,cls(i)), x, sigma);
    if cls(i) == y
        df(:,:,i) = 2*(rbf_1(i) - 1).*rbf_b1 + 2*(rbf_2(i) - 1).*rbf_b2';
    else
        df(:,:,i) = (2*beta/(N-1)).*(rbf_1(i).*rbf_b1 + rbf_2(i).*rbf_b2');
    end
end
df = sum(df, 3);
end


