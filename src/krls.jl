function krls(x, y; nu=1., lambda=0.1, kernelfunc=linear_kernel, maxdict=100, prec=float32, index="lin")

# krls(x, y)
#
# Julia version of the Kernel Recursive Least Squares Algorithm
#
# Input:
# 			  x : N(samples) x d(dimension) matrix
# 			  y : N(samples) x 1 vector
# 	         nu : approximate linear dependency (ALD) threshold, controls sparsity (default: 1.)
#        kernel : type of kernel (default : 'linear_kernel')
#       maxdict : maximum dictionary size (default: 100)
#         index : how to cycle through samples, 'lin' goes in order, 'rand' goes randomly, or you can provide your own vector index (default: 'lin')
#
# Output:
#		  alpha : weights on dictionary samples
#		   dict : dictionary samples
#		   Kinv : Inverse of kernel matrix
#	   dict_idx : dictionary sample indicies
#

# nu=1.; lambda=0.1; kernelfunc=linear_kernel; maxdict=100; prec=float32; index="lin"

x = prec(x);
y = prec(y);
lambda = prec(lambda);
nu = prec(nu);

lambda2 = lambda.^2;

sz = size(x);

dict = zeros(eltype(x), sz[1], maxdict);
dict_idx = zeros(eltype(x), maxdict, 1);
## Set order to look at points
if index == "rand"
	idx = randperm(sz[2]);
elseif index == "lin"
	idx = 1:sz[2];
else
	idx = index;
end

## Initialize
K = kernelfunc(x[:,idx[1]], x[:,idx[1]]) + lambda2;
Kinv = (1./K)';
alpha = (y[idx[1]]./K)';
dict[:,1] = x[:,idx[1]];
dict_idx[1] = idx[1];
P = 1;
m = 1;
m2 = 1;

for ii = idx[2:end];

	m2 = m2 + 1;

	kt = kernelfunc(dict[:,1:m], x[:,ii]) + lambda2;

	ktt = kernelfunc(x[:,ii], x[:,ii]) + lambda2;

	at = Kinv * kt';

	dt = ktt - kt*at;

	if abs(dt[1]) > nu && m < maxdict

		m = m + 1;

		dict[:,m] = x[:,ii];
		dict_idx[m] = ii;

		Kinv = (1 ./ dt) .* [dt.*Kinv + at*at' -at; -at' 1];

		P = [P zeros(eltype(x),size(P,1), 1); zeros(eltype(x),1, size(P,1)) 1];

		alpha = [alpha - ((at ./ dt) * (y[ii] - kt*alpha)); (1./dt)*(y[ii] - kt*alpha)];

		if mod(m,50)==0
			println("Dictionary Size: $m of $maxdict")
		end
		
	else

		Pat = P*at;
		atPat = 1 + at'*Pat;

		qt = Pat / atPat[1];

		atP = (at'*P);

		# P -= ((Pat*(at'*P)) ./ atPat);
		# optimized as
		Base.LinAlg.BLAS.gemm!('N', 'N', -one(eltype(y)), qt,atP, one(eltype(y)), P);

		# alpha +=  Kinv*qt*(y[ii] - kt*alpha);
		# optimized as
		kta = kt*alpha;

		dif = y[ii] - kta[1];

		Base.LinAlg.BLAS.gemm!('N', 'N', dif, Kinv,qt, one(eltype(y)), alpha);

		if mod(m2,50)==0
			println("On sample: $m2 of $(sz[2])")
		end

	end

end

dict = dict[:,1:m];
dict_idx = dict_idx[1:m];

(alpha, dict, Kinv, dict_idx)

end