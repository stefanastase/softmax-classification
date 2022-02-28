function Z = softmax(Z)
    [n,m] = size(Z);
    for i=1:n
        sum_exp = sum(exp(Z(i,:)));
        for j=1:m
            Z(i, j) = exp(Z(i,j))/sum_exp;
        end
    end
end

