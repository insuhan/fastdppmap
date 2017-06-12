function X = greedy_alg1_constrained(A,K, ncluster, nrank)
% This function is implemented based on lazy-evaluations (Minoux, 1978).
N = size(A,1);
cg_iter = 30;

dA = diag(A)';
deltas = dA;
Y = 1:N;
argmax = find(deltas == max(deltas), 1);
X = Y(argmax);
Y(argmax) = [];
deltas(argmax) = [];
Es = [A(X,Y); dA(Y)];

cnt = 0;
tot = 0;
while(1)
    try 
        [AX_inv_cols, IDX, C, ld_cluster] = logdet_grad(A(X,X), Es, ncluster, cg_iter);
    catch 
        fprintf('sfd\n');
    end
    
    bestimprov = 0;
    [~,order] = sort(deltas,'descend');        
    for test = order
        if ismember(Y(test), X)
            error('Number of Y does not match!');
        end
        if deltas(test) >= bestimprov
            tmp = AX_inv_cols(:,IDX(test)).*(Es(:,test) - C(:,IDX(test)));
            improv = 2*sum(tmp) - tmp(end) + ld_cluster(IDX(test));
            cnt = cnt + 1;
            deltas(test) = improv;
            bestimprov = max(bestimprov, improv);
        elseif deltas(test) > -inf
            break;
        end
    end
    tot = tot + size(order,2);
    
    [~,sorted_idx] = sort(deltas,'descend');
    deltas_sel = zeros(1,nrank);
    for k = 1 : nrank
        idx = sorted_idx(k);
        deltas_sel(k) = logdet_margin_cg(A,X,Y(idx),cg_iter);
        deltas(idx) = deltas_sel(k);
    end
    
    [dmax, sel_argmax] = max(deltas_sel);
    argmax = sorted_idx(sel_argmax);
    if dmax < 0 || length(X) == K
        break;
    else
        Es(end,:) = A(Y(argmax),Y);
        Es = [Es; dA(Y)];
        Es(:,argmax) = [];
        
        X = [X Y(argmax)];
        Y(argmax) = [];
        deltas(argmax) = []; 
    end
end
end

function [AX_inv_cols, IDX, C, ld_cluster] = logdet_grad(AX,Es,ncluster,cg_iter)
NX = size(AX,1);
AX_mean = zeros(NX+1,NX+1);
AX_mean(1:NX,1:NX) = AX;

[IDX,C] = random_clustering(Es,ncluster);
e_end = sparse(NX+1,1,1,NX+1,1);
AX_inv_cols = zeros(NX+1,ncluster);
ld_cluster = zeros(1,ncluster);
for k = 1 : ncluster
    AX_mean(:,end) = C(:,k);
    AX_mean(end,:) = C(:,k)';
    
    AX_inv_cols(:,k) = cg_linear_solver(AX_mean,e_end,cg_iter);
    
    ux = C(1:end-1,k);
    x = cg_linear_solver(AX,ux,cg_iter);
    ld_cluster(k) = log(C(end,k) - sum(ux.*x));
end
end

function [IDX,C] = random_clustering(Es,ncluster)
NY = size(Es,2);
IDX = mod(randperm(NY),ncluster) + 1;
C = zeros(size(Es,1),ncluster);
for i = 1 : ncluster
    C(:,i) = mean(Es(:,IDX==i),2);
end
end

