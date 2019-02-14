function RichardsonExtrap(N, lb = -3, p = 2, q = 2, l = 1, x0 = 0, T = 1)
M = zeros(N, N)
	function fn(n)
		return 1/n^l
	end
	function A(n)
		return BCP(n, fn(n), T, x0, lb)
	end
M[1, 1] = A(2^0)
for j = 2:N
	M[j,1] = A(q^(j-1))
	for k = 2:j
		M[j, k] = (q^((k-1)*p)*M[j, k-1] - M[j-1, k-1])/(q^((k-1)*p)-1)
	end
end
return M
end

