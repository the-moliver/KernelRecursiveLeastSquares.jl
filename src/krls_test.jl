using KernelLibrary, KernelRecursiveLeastSquares, ProfileView

x = randn(10000,50);
h = randn(50,1);
h2 = randn(50,1);
y = (x*h + 1).^3 + 4*(x*h2 + 1).^2+ 200*randn(10000,1);
stdy = std(y)
y ./= stdy;

x2 = randn(200,50);
y2 = (x2*h + 1).^3 + 4*(x2*h2 + 1).^2;
y2 ./= stdy;

x=x'; y=y';
x2=x2'; y2=y2';


k(x,y) = polynomial_kernel(x,y,3)

@time (alpha1, dict1, Kinv1, dict1_idx) = krls(x, y, kernelfunc=k, nu=1., maxdict=200);

# @profile (alpha1, dict1, Kinv1, dict1_idx) = krls(x, y, kernelfunc=k, nu=1., maxdict=100);

profileView.view()

ypred = alpha1'*k(dict1, x2);

cor(ypred', y2')

k(x,y) = multiquad_kernel(x,y,1)';

@time (alpha1, dict1, Kinv1, dict1_idx) = krls(x, y, kernelfunc=k, nu=.5, maxdict=200);

ypred2 = zeros(size(x2,2),1);
for ii in [1:size(x2,2)]
	tmp = alpha1'*k(dict1, x2[:,ii]);
	ypred2[ii]=tmp[1];
end

cor(ypred2, y2')

k(x,y) = power_kernel(x,y,3)';

@time (alpha1, dict1, Kinv1, dict1_idx) = krls(x, y, kernelfunc=k, nu=.5, maxdict=200);

ypred2 = zeros(size(x2,2),1);
for ii in [1:size(x2,2)]
	tmp = alpha1'*k(dict1, x2[:,ii]);
	ypred2[ii]=tmp[1];
end

cor(ypred2, y2')