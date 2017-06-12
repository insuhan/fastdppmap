function u = cg_linear_solver(A,b,max_iter)
u = b;
r = b - A * u;
p = r;
for m = 1 : max_iter
    if(norm(r) < 1e-15)
        break;
    end
    a = A * p;
    a_dot_p = a' * p;
    lambda = (r' * p) / a_dot_p;
    u = u + lambda * p;
    r = r - lambda * a;
    p = r - ((r' * a) / a_dot_p) * p;
end
end
