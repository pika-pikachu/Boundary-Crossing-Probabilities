# RW
using PyPlot
using Distributions

function g(t)
    return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

function bbb(x0, x1, t0, t1)
    return 1 - exp(-2(g(t1) - x1)*(g(t0)-x0)/(t1-t0))
end

function absorb2(n, L=-3)
if g(1/n) < 1/sqrt(n) # 
    
    # state space
    ss = [-1] 

    # transition probabilities
    p = [bbb(0,-1/sqrt(n), 0, 1/n)*1/2]
else 

    # state space, one jump up or down
    ss = [-1 1]

    p = [bbb(0,-1/sqrt(n), 0, 1/n)*1/2 bbb(0, 1/sqrt(n), 0, 1/n)*1/2]
end
    for i = 2:n

        # new state space
        ss_new = ( ss[1] - 1 ):2:min( g(i/n)*sqrt(n), ss[end] + 1 )
        M = zeros(length(ss), length(ss_new))
        N = min(length(ss), length(ss_new))
        N2 = size(M)
        for j = 1:(N-1)
            for k = 1:length(ss_new)
                if j == k || k == j + 1
                    M[j,k] = bbb(ss[j]/sqrt(n), ss_new[k]/sqrt(n), (i-1)/n, i/n)/2
                end
            end
        end
        M[N,N] = bbb(ss[N]/sqrt(n), ss_new[N]/sqrt(n), (i-1)/n, i/n)/2
        M[N2[1],N2[2]] = bbb(ss[N2[1]]/sqrt(n), ss_new[N2[2]]/sqrt(n), (i-1)/n, i/n)/2    
        p = p*M
        ss = ss_new # New state space becomes old
    end
return p
end

function guideplot(N, s = 0.05)
    plt[:yscale]("log")
    grid("on")
    plot(1:N, 1 ./ ((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
    plot(1:N, 1 ./ (1:N)*s ,       label = L"1/n",     ls = "--", color = "black")
    plot(1:N, 1 ./ ((1:N).^2)*s,   label = L"1/n^2",   ls = "--", color = "black")
    plot(1:N, 1 ./ ((1:N).^4)*s,   label = L"1/n^4",   ls=  "--", color = "black")
end

function conv2(N,sp = 10)
mesh = 1:sp:N
m = length(mesh)
z1 = zeros(m)
d = 0.4797493549688766
for i = 1:m
    z1[i] = abs( (1 - sum(absorb2(floor(Int, mesh[i]) )) ) - d )
end
lw = 1.5
marker = "D"
figure()
guideplot(N,0.2)
plt[:xscale]("log")
plt[:yscale]("log")
plot(mesh, z1, marker=marker, lw = lw, color = "black", label = "w/bb")
legend()
end
conv2(100,10)









