function krls{T}(x::Array{T}, y::Array{T}; nu=1., λ=0.1, kernelfunc=linear_kernel, maxdict=100, index="lin")

# krls(x, y)
#
# Julia version of the Kernel Recursive Least Squares Algorithm
#
# Input:
# 			  x : d(dimension) x N(samples) array
# 			  y : 1 x N(samples) vector
# 	         nu : approximate linear dependency (ALD) threshold, controls sparsity (default: 1.)
# 	     λ : regularization parameter (default: 0.1)
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
#
# Citation:
# Y. Engel, S. Mannor, and R. Meir, “The kernel recursive least-squares algorithm,” IEEE Transactions on Signal Processing, vol. 52, no. 8, pp. 2275–2285, 2004.


λ = convert(T, λ)
nu = convert(T, nu)

λ2 = λ.^2

sz = size(x)

dict = zeros(eltype(x), sz[1], maxdict)
dict_idx = zeros(eltype(x), maxdict, 1)
## Set order to look at points
if index == "rand"
	idx = randperm(sz[2])
elseif index == "lin"
	idx = 1:sz[2];
else
	idx = index;
end

## Initialize
K = kernelfunc(x[:,idx[1]], x[:,idx[1]]) + λ2
Kinv = (1./K)'
alpha = (y[idx[1]]./K)'
dict[:,1] = x[:,idx[1]]
dict_idx[1] = idx[1]
P = one(eltype(x))
m = 1
m2 = 1

for ii = idx[2:end]

	m2 = m2 + 1

	kt = kernelfunc(dict[:,1:m], x[:,ii]) + λ2

	ktt = kernelfunc(x[:,ii], x[:,ii]) + λ2

	at = Kinv * kt'

	dt = ktt - kt*at

	kta = kt*alpha

	e = y[ii] - kta[1]

	if abs(dt[1]) > nu && m < maxdict

		m = m + 1

		dict[:,m] = x[:,ii]
		dict_idx[m] = ii

		Kinv = (1 ./ dt) .* [dt.*Kinv + at*at' -at; -at' 1]

		P = [P zeros(eltype(x),size(P,1), 1); zeros(eltype(x),1, size(P,1)) 1]

		alpha = [alpha - ((at ./ dt) * e); (1./dt)*e]

		if mod(m,50)==0
			println("Dictionary Size: $m of $maxdict")
		end
		
	else

		Pat = P*at
		atPat = 1 + at'*Pat

		qt = Pat / atPat[1]

		atP = (at'*P)

		# P -= ((Pat*(at'*P)) ./ atPat)
		# optimized as
		Base.LinAlg.BLAS.gemm!('N', 'N', -one(eltype(y)), qt,atP, one(eltype(y)), P)

		# alpha +=  Kinv*qt*(y[ii] - kt*alpha)
		# optimized as

		Base.LinAlg.BLAS.gemm!('N', 'N', e, Kinv,qt, one(eltype(y)), alpha)

		if mod(m2,50)==0
			println("On sample: $m2 of $(sz[2])")
		end

	end

end

dict = dict[:,1:m]
dict_idx = dict_idx[1:m]

(alpha, dict, Kinv, dict_idx)

end