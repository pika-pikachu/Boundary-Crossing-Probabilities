# Markov chain approximation of Integrated Brownian motion
# August 10th 2019
#
# Uses the exact transition probabiltiy of (W_t, I_t). This version is for a one sided boundary
# and the grid is boundary dependent to ensure smooth convergence.

using TensorOperations
using Plots

function g(t)
	return 0.4
end

@doc """
	transprob_IBM(x, y, dt, h)

Returns the transition probability of the integrate Brownian motion
x: starting position
y: ending position
dt: time step size
h: space step size
""" -> 
function transprob_IBM(x0, y0, x1, y1, dt, h)
x = x1 - x0
y = y1 - y0 - x0*dt
dx = h
dy = h*dt
	return sqrt(3)/(pi*dt^2)*exp(-2/dt^3*(dt^2*x^2 - 3*dt*x*y + 3*y^2))*dx*dy
end


# currently broken fix this
function absorb(x0, y0, x1, dt, h, L)
x = x1 - x0
	return exp(-x^2/(2*dt))/sqrt(2*pi*dt)*cdf(Normal(), sqrt(3)*(-2*L + dt*x + 2*dt*x0 + 2*y0)/(dt^(3/2)))*h 
end	

@doc """
	pmatrix01(n, h, x_inf, y_inf)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time 1/n
n: number of time partitions
h: space step size
""" ->
function pmatrix01(n::Int, h, x_inf = 3, y_domain = [-1.75,2])
U_x = floor(Int, x_inf/h)
L_x = ceil(Int, -x_inf/h)
x_1_mesh = U_x:-1:L_x
y_1_mesh = g(1/n):-h/n:y_domain[1]
M = zeros(length(x_1_mesh), length(y_1_mesh))
	for i = 1:length(x_1_mesh)
		for j = 1:(length(y_1_mesh))
			if y_1_mesh[j] < g(1/n)
				M[i, j] = transprob_IBM(0, 0, h*x_1_mesh[i], y_1_mesh[j], 1/n, h)
			end
		end
		# M[i,length(y_1_mesh)] = absorb(0, 0, h*x_1_mesh[i], 1/n, h, y_domain[1])
	end
return M
end


@doc """
	pmatrix_i(n, h, x_inf, y_inf)

Transition from (x_i, y_i) to (x_{i+1}, y_{i+1}). 
n: number of time partitions
h: space step size
""" -> 
function pmatrix_i(t_i, n::Int, h, x_inf = 3, y_domain = [-1.75,2])
U_x = floor(Int, x_inf/h)
L_x = ceil(Int,-x_inf/h)
x_0_mesh = U_x:-1:L_x 
x_1_mesh = U_x:-1:L_x
y_0_mesh = g(t_i/n):-h/n:y_domain[1]
y_1_mesh = g((t_i+1)/n):-h/n:y_domain[1]
M = zeros(length(x_0_mesh),length(y_0_mesh),length(x_1_mesh),length(y_1_mesh))
	for i = 1:length(x_0_mesh)
		for k = 1:length(x_1_mesh)
			for j = 1:(length(y_0_mesh))
				for l = 1:(length(y_1_mesh))
					if (y_0_mesh[j] < g(t_i/n)) && (y_1_mesh[l] < g(t_i/n))
						M[i,j,k,l] = transprob_IBM(h*x_0_mesh[i], y_0_mesh[j], h*x_1_mesh[k], y_1_mesh[l], 1/n, h)
					end
				end
				# M[i,j,k,length(y_1_mesh)] = absorb(h*x_0_mesh[i],y_0_mesh[j],h*x_1_mesh[k],1/n,h,y_domain[1])
			end			
		end
	end
M[length(x_0_mesh),length(y_0_mesh),length(x_1_mesh),length(y_1_mesh)] = 1
return M
end

function anti_transpose(M)
	return M[size(M)[1]:-1:1, size(M)[2]:-1:1]'
end


@doc """
	t_mat(n, h, x_inf, y_domain, fps, anim_plot)

Returns the surviving transition matrix
t_mat(10, 1/10^0.6, 3, [-2,2], 10, 1)
t_mat(5, 1/5^0.6, 3, [-2,2], 10, 0)

gr()
plotlyjs()
M1 = t_mat(6,1/6^0.6)
M2 = t_mat(8,1/8^0.6)
plot(heatmap(anti_transpose(M1), heatmap(anti_transpose(M2))
""" -> 
function t_mat(n::Int, h, x_inf = 3, y_domain = [-2,2], fps = 5, anim_plot = 0)
P_old = pmatrix01(n, h, x_inf, y_domain) # step from 0 -> 1 -> 2
	for i in 1:(n-1)
		Mi = pmatrix_i(i, n, h, x_inf, y_domain)
		P_new = zeros(size(Mi)[1], size(Mi)[2])
		@tensor begin 
			P_new[x1, y1] = P_old[x0, y0]*Mi[x0, y0, x1, y1]
		end
		P_old = P_new
	end
	return P_old
end

# function guideplot(N, s = 0.05)
# 	pvec = [0.5, 1, 1.5, 2, 3, 4]
# 	plot(1:N, 1 ./( (1:N).^(pvec[1]) )*s, xaxis=:log, yaxis=:log)
# 	for i in 2:length(pvec)
# 		plot!(1:N, 1 ./( (1:N).^(pvec[i]) )*s, xaxis=:log, yaxis=:log)
# 	end	
# end

function guideplot(N, s = 0.05)
	# plt.xscale("log")
	# plt.yscale("log")
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	pvec = [0.5, 1, 1.5, 2, 3, 4]
	for i in 1:length(pvec)
		plot(1:N, 1 ./( (1:N).^(pvec[i]) )*s, ls = "--", color = "black")
		plt[:text](N + 0.3, 0.5*s/N^(pvec[i]) , join(["n^-", pvec[i]]), fontsize = 9)
	end	
	xticks(unique(vcat(1:10, 10:10:100, 100:100:1000, 1000:1000:10000)))
end



function convergence(N = 10, p=1.0, γ = 1, x_inf = 3, y_domain = [-1.75,g(1)])
p_vec = zeros(N)
for n in 1:N
	h = γ*n^(-p)	
	M = t_mat(n, h, x_inf, y_domain)
	p_vec[n] = sum(M)
end
p_diff = abs.(p_vec[2:end] - p_vec[1:(end-1)])
# plot(1:(N-1), p_diff, xaxis=:log, yaxis=:log)
plt[:xscale]("log")
plt[:yscale]("log")
plot(1:(N-1), p_diff)
end

convergence(10,0.6)
guideplot(10,0.3)



N = 10 
p = 1.0
γ = 1 
x_inf = 3 
y_domain = [-2,g(1)]

p_vec = zeros(N)
for n in 1:N
	h = γ*n^(-p)	
	M = t_mat(n, h, x_inf, y_domain)
	p_vec[n] = sum(M)
end
p_diff = abs.(p_vec[2:end] - p_vec[1:(end-1)])
plot(1:(N-1), p_diff, xaxis=:log, yaxis=:log)

plot!(1:(N-1), p_diff, xaxis=:log, yaxis=:log)


plot!(1:N, 1 ./( (1:N).^(2) ), xaxis=:log, yaxis=:log)


plot!(1:N, 1 ./( (1:N).^(2.75) ), xaxis=:log, yaxis=:log)

# figure()
# plt[:imshow](P_old, extent = [-3,3,-3,3], cmap = "afmhot", alpha=1, aspect="auto", interpolation="none")
# title(L"Heat map $ X_t = (W_t, \int_0^t W_u du) $")
# xlabel(L"$W_t$")
# ylabel(L"$\int_0^t W_u du$")