function X = greedy_alg2(A,batch,nsample,ncluster,nrank,emax,emin,m,n)
N = size(A,1);
cg_iter = 30;

X = greedy_alg1_constrained(A,batch,ncluster,nrank);
Y = setdiff(1:N,X);
ldX = log(det(A(X,X)));
dA = diag(A);

for nt = 1 : N
    Es = [A(X,Y); dA(Y)'];
    if ncluster == 1
        val1_hat = logdet_diff_mean(A(X,X),Es,cg_iter);        
    else
        val1_hat = logdet_diff_kmeans(A(X,X),Es,ncluster,cg_iter);        
    end
    
    [~,idx] = sort(val1_hat,'descend');
    val1 = zeros(1,nrank);
    Ytop = Y(idx(1:nrank));
    for i = 1 : nrank
        Xii = [X Ytop(i)];
        val1(i) = logdet(A(Xii,Xii));
    end
    [ldX_new_single, max_idx] = max(val1);
    max_idx_single = find(Y==Ytop(max_idx));
    
    sets = zeros(nsample,batch);
    Es_batch = zeros(length(X) + batch, batch, nsample);
    for j = 1 : nsample
        sets(j,:) = randperm(length(val1),batch);
        Ysample = Ytop(sets(j,:));
        Es_batch(:,:,j) = A([X Ysample], Ysample);
    end
    
    if ncluster == 1
        val2_hat = logdet_diff_mean_batch(A(X,X),Es_batch,batch);
    else
        val2_hat = logdet_diff_kmean_batch_ld(A(X,X),Es_batch,ncluster,batch,cg_iter,emax,emin,m,n);
    end
    [~,idx2] = sort(val2_hat,'descend');
    sets2 = sets(idx2(1:nrank),:);
    val2 = zeros(1,nrank);
    for j = 1 : length(sets2)
        Ysample = Ytop(sets2(j,:));
        Xii = [X Ysample];
        val2(j) = logdet(A(Xii,Xii));
    end
    [ldX_new_batch,max_idx] = max(val2);
    
    if ldX_new_batch > ldX
        YsampleBest = Ytop(sets2(max_idx,:));
        X = [X YsampleBest];
        ldX = ldX_new_batch;
        Y = setdiff(Y, YsampleBest);
    elseif ldX_new_single > ldX
        X = [X Y(max_idx_single)];
        ldX = ldX_new_single;
        Y = setdiff(Y, Y(max_idx_single));
    else
        break;
    end
    
end

end

function val = logdet_diff_mean(AX,Es,cg_iter)
NX = size(AX,1);
NY = size(Es,2);
val = zeros(1,NY);

Es_mean = mean(Es,2);

AX_mean = zeros(NX+1,NX+1);
AX_mean(1:NX,1:NX) = AX;
AX_mean(:,end) = Es_mean;
AX_mean(end,:) = Es_mean';

e_end = sparse(NX+1,1,1,NX+1,1);
AX_inv_cols = cg_linear_solver(AX_mean,e_end,cg_iter);

for i = 1 : NY
    tmp = AX_inv_cols.*(Es(:,i) - Es_mean);
    val(i) = 2*sum(tmp) - tmp(end);
end
end

function val = logdet_diff_kmeans(AX,Es,ncluster,cg_iter)
NX = size(AX,1);
NY = size(Es,2);
val = zeros(1,NY);

AX_mean = zeros(NX+1,NX+1);
AX_mean(1:NX,1:NX) = AX;
[IDX,C] = random_clustering(Es,ncluster);
% opts = statset('MaxIter',10);
% [IDX,C] = kmeans(full(Es)',ncluster,'Start','sample','Replicates',1,'Options',opts);
% C = C';

e_end = sparse(NX+1,1,1,NX+1,1);
AX_inv_cols = zeros(NX+1,ncluster);
ld_cluster = zeros(1,ncluster);

for k = 1 : ncluster
    AX_mean(:,end) = C(:,k);
    AX_mean(end,:) = C(:,k)';
    ld_cluster(k) = logdet(AX_mean);
    AX_inv_cols(:,k) = cg_linear_solver(AX_mean,e_end,cg_iter);
end

for i = 1 : NY
    tmp = AX_inv_cols(:,IDX(i)).*(Es(:,i) - C(:,IDX(i)));
    val(i) = 2*sum(tmp) - tmp(end) + ld_cluster(IDX(i));
end
end

function [IDX,C] = random_clustering(Es,ncluster)
NY = size(Es,2);
rng(1234);
IDX = mod(randperm(NY),ncluster) + 1;
C = zeros(size(Es,1),ncluster);
for i = 1 : ncluster
    C(:,i) = mean(Es(:,IDX==i),2);
end
end

function val = logdet_diff_mean_batch(AX,Es,b)
NX = size(AX,1);
NY = size(Es,3);
Es_mean = mean(Es,3);

AX_mean = zeros(NX+b,NX+b);
AX_mean(1:NX,1:NX) = AX;
AX_mean(:,NX+1:end) = Es_mean;
AX_mean(NX+1:end,:) = Es_mean';

e_end_batch = sparse((NX+1:NX+b)',(1:b)',ones(b,1),NX+b,b);
AX_inv_cols_batch = AX_mean \ e_end_batch;

val = zeros(1,NY);
for i = 1 : NY
    col_prod_sum = AX_inv_cols_batch.*(Es(:,:,i) - Es_mean);
    dup_part = col_prod_sum(NX+1:NX+b,:);
    val(i) = 2*sum(col_prod_sum(:)) - sum(dup_part(:));    
end
end

function val = logdet_diff_kmean_batch_ld(AX,Es,ncluster,b,cg_iter,emax,emin,m,n)
NX = size(AX,1);
NY = size(Es,3);
assert(NY >= ncluster);

AX_mean = zeros(NX+b,NX+b);
AX_mean(1:NX,1:NX) = AX;

Es_vec = reshape(Es, size(Es,1) * size(Es,2), size(Es,3));
[IDX,C] = random_clustering(Es_vec,ncluster);

e_end_batch = sparse((NX+1:NX+b)',(1:b)',ones(b,1),NX+b,b);
AX_inv_cols = zeros(NX+b,b,ncluster);
ld_cluster = zeros(1,ncluster);

vv = sign(randn(NX+b,m));

%%%% cheby version %%%%
for k = 1 : ncluster
    tmp = reshape(C(:,k),size(Es,1),size(Es,2));
    AX_mean(:,NX+1:end) = tmp;
    AX_mean(NX+1:end,:) = tmp';
    for j = 1 : b
        AX_inv_cols(:,j,k) = cg_linear_solver(AX_mean,e_end_batch(:,j),cg_iter);
    end
    ld_cluster(k) = cheb_logdet(AX_mean,emin,emax,m,n,vv);
end

val = zeros(1,NY);
for i = 1 : NY
    tmp = reshape(C(:,IDX(i)), size(Es,1), size(Es,2));
    col_prod_sum = AX_inv_cols(:,:,IDX(i)).*(Es(:,:,i) - tmp);
    dup_part = col_prod_sum(NX+1:NX+b,:);
    val(i) = 2*sum(col_prod_sum(:)) - sum(dup_part(:)) + ld_cluster(IDX(i));    
end
end

function ld = cheb_logdet(AX,emin,emax,m,n,vv)
delta = emin / (emin + emax);
r = 2/(1-2*delta);
b = -1/(1-2*delta);

AX_hat = (AX/(emin+emax)).*r + b*speye(size(AX));

f = @(x) log(x);
g = @(x) ((1-2*delta)/2).*x+0.5;
h = @(x) f(g(x));
c = chebpolfit(h,n);

w0 = vv;
w1 = AX_hat*vv;
u = c(1)*w0 + c(2)*w1;
for j = 2 : n
    ww = 2*(AX_hat)*w1 - w0;
    u = u + c(j+1)*ww;
    w0 = w1;
    w1 = ww;
end
ld = sum(sum(u.*vv))/m + size(AX,1)*log(emin + emax);
end