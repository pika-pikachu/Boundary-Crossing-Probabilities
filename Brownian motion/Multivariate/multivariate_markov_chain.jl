function p1(n::Int, h, m_inf = 3)
x_mesh = m_inf:(-h):(-m_inf) # mesh in terms of BM's mesh
y_mesh = m_inf:(-h):(-m_inf)
vec = zeros(length(x_mesh), length(y_mesh))
	for j = 1:length(x_mesh)
		for k = 1:length(y_mesh)
			vec[j, k] = transprob(0, x_mesh[j], 1/n, h)*transprob(0, y_mesh[k], 1/n, h)
		end
	end
return vec
end


function p_i(z, n::Int, h, m_inf = 3)
x_0_mesh = m_inf:(-h):(-m_inf) # mesh in terms of BM's mesh
y_0_mesh = m_inf:(-h):(-m_inf) # mesh in terms of BM's mesh
x_1_mesh = m_inf:(-h):(-m_inf) # mesh in terms of BM's mesh
y_1_mesh = m_inf:(-h):(-m_inf) # mesh in terms of BM's mesh
p_mat = zeros(length(x_0_mesh), length(x_1_mesh), length(y_0_mesh), length(y_1_mesh))
for i = 1:length(x_0_mesh)
	for j = 1:length(y_0_mesh)
		for k = 1:length(x_1_mesh)
			for l = 1:length(y_1_mesh)
				p_mat[i, j, k, l] = transprob(x_0_mesh[i], x_1_mesh[k], 1/n, h)*transprob(y_0_mesh[j], y_1_mesh[l], 1/n, h)
			end
		end
	end
end
return p_mat
end

n = 10

P1 = p1(n,1/sqrt(n))
P2 = p_i(n,1/sqrt(n))

pcolormesh(P1, vmin = 0, vmax = 0.02)

@einsum Pn[k,l] := P1[i,j] * P2[i,j,k,l]
pcolormesh(Pn, vmin = 0, vmax = 0.02)