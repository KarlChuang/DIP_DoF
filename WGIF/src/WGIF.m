A = imread("../image/0815_depth2.png");
B = imread("../image/0815.png");

Iguided = B;
Iguidedr = Iguided(:, :, 1);
Iguidedg = Iguided(:, :, 2);
Iguidedb = Iguided(:, :, 3);
for i= 1:7
    Iguidedr = weightedGuidedImageFilter(Iguidedr, Iguidedr, 16, 0.005);
    Iguidedg = weightedGuidedImageFilter(Iguidedg, Iguidedg, 16, 0.005);
    Iguidedb = weightedGuidedImageFilter(Iguidedb, Iguidedb, 16, 0.005);
end
Z_GIF = zeros(size(B));
Z_GIF(:,:,1) = Iguidedr;
Z_GIF(:,:,2) = Iguidedg;
Z_GIF(:,:,3) = Iguidedb;

res = weightedGuidedImageFilter(A, rgb2gray(Z_GIF), 16, 0.005);

subplot(2,2,1),
imshow(B);
subplot(2,2,2),
imshow(Z_GIF);
subplot(2,2,3),
imshow(A);
subplot(2,2,4),
imshow(res);

imwrite(res, '0815_result.jpg')



