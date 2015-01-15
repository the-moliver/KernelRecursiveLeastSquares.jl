using KernelLibrary, KernelRecursiveLeastSquares, Winston

x = randn(5000,20);
h = randn(20,1);
h2 = randn(20,1);
y = (x*h + 1).^3 + 4*(x*h2 + 1).^2+ 500*randn(5000,1);

x2 = randn(200,20);
y2 = (x2*h + 1).^3 + 4*(x2*h2 + 1).^2;

x=float32(x'); y=float32(y');
x2=float32(x2'); y2=float32(y2');


k(x,y) = polynomial_kernel(x,y,2);

(alpha1, dict1, Kinv1, dict1_idx) = krls(x, y, kernelfunc=k, nu=1., maxdict=30);

@time (alpha1, dict1, Kinv1, dict1_idx) = krls(x, y, kernelfunc=k, nu=1., maxdict=300);


ypred1 = zeros(Float32, size(x2,2),1);
for ii in [1:size(x2,2)]
	tmp = k(dict1, x2[:,ii])*alpha1;
	ypred1[ii]=tmp[1];
end

cor(ypred1, y2')

k(x,y) = multiquad_kernel(x,y,1);

@time (alpha2, dict2, Kinv2, dict2_idx) = krls(x, y, kernelfunc=k, nu=3., maxdict=300, index="rand");

ypred2 = zeros(Float32, size(x2,2),1);
for ii in [1:size(x2,2)]
	tmp = k(dict2, x2[:,ii])*alpha2;
	ypred2[ii]=tmp[1];
end

cor(ypred2, y2')

k(x,y) = power_kernel(x,y,3);

@time (alpha3, dict3, Kinv3, dict3_idx) = krls(x, y, kernelfunc=k, nu=.5, maxdict=300, index="rand");

ypred3 = zeros(Float32, size(x2,2),1);
for ii in [1:size(x2,2)]
	tmp = k(dict3, x2[:,ii])*alpha3;
	ypred3[ii]=tmp[1];
end

cor(ypred3, y2')

hold(false)
plot(y2')
hold(true)
plot(ypred1, "r");
plot(ypred2, "g")
plot(ypred3, "b")


mp = mean([ypred1 ypred2 ypred3],2);
plot(mp,"m")
cor(mp, y2')