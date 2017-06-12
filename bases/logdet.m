function ld = logdet(A)
ld = 2*sum(log(diag(chol(A)))) + 0;
end