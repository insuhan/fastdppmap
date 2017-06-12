function y = logdet_margin_cg(A,X,i,cg_iter)
ux = A(X,i);
x = cg_linear_solver(A(X,X),ux,cg_iter);
% [x,~] = pcg(A(X,X),ux,1e-10);
y = log(A(i,i) - sum(ux.*x));
end