function Z = weightedGuidedImageFilter(X, G, r, lambda, gamma)
% X:        input image, must be a gray-scale image
% G:        guidance image, must be a gray-scale image
% r:        window radius
% lambda:   regularization parameter
% gamma:    selective
% Z:        filter output

if(exist('gamma_G', 'var') ~= 1)
    gamma = edgeAwareWeighting(G);
end

X = im2double(X);
G = im2double(G);

[hei, wid] = size(X);
N = boxFilter(ones(hei, wid), r);

GX = G .* X;
mean_GX = boxFilter(GX, r) ./ N;
mean_X = boxFilter(X, r) ./ N;
mean_G = boxFilter(G, r) ./ N;

mean_GG = boxFilter(G.*G, r) ./ N;
var_G = mean_GG - mean_G .* mean_G;

a = (mean_GX - mean_G.*mean_X) ./ (var_G + lambda./gamma);
b = mean_X - a .* mean_G;

mean_a = boxFilter(a, r) ./ N;
mean_b = boxFilter(b, r) ./ N;

Z = mean_a .* G + mean_b;       % Eqn(9) in the paper

end