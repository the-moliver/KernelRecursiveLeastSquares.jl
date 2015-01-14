x = randn(5000,20);
h = randn(20,1);
h2 = randn(20,1);
y = (x*h + 1).^3 + 4*(x*h2 + 1).^2+ 500*randn(5000,1);

x2 = randn(200,20);
y2 = (x2*h + 1).^3 + 4*(x2*h2 + 1).^2;


kernelfunc(x,y) = polynomial_kernel(x,y,2);

[alpha1, dict1, Kinv1, dict1_idx] = KRLS(x, y, kernelfunc)

ypred = ((dict1*x2' + 1) .^ 2)'*alpha1;

cor(ypred, y2)

