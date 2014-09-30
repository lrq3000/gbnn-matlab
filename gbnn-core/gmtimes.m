function C = gmtimes(A, B, outop, inop, Csize)
% C = gmtimes(A, B, outop, inop, Csize)
% Generalized matrix multiplication between A and B. By default, standard sum-of-products matrix multiplication is operated, but you can change the two operators (inop being the element-wise product and outop the sum).
% Speed note: about 100-200x slower than A*A' and about 3x slower when A is sparse, so use this function only if you want to use a different set of inop/outop than the standard matrix multiplication.
% TODO: slower on octave because bsxfun is not sparse aware, speed can potentially be hugely impacted if we use a sparse-aware bsxfun like in MatLab!

if ~exist('inop', 'var')
    inop = @times;
end

if ~exist('outop', 'var')
    outop = @sum;
end

if ~exist('B', 'var')
    B = [];
end

if ~exist('Csize', 'var')
    Csize = [];
end

[n, m] = size(A);
[m2, o] = size(B);

symmetry_flag = false;
if isempty(B) % if size(A) == size(B') && A == B' % optimization: recopy half lower of the matrix since it's symmetrical
    symmetry_flag = true;
    m2 = m;
    o = n;
    B = A;
end
if m2 ~= m
    error('genmtimes.m: nonconformant arguments (op1 is %ix%i, op2 is %ix%i).', n, m, m2, o);
end
if ~isempty(Csize) && numel(Csize) == 2 % User can specify the output's size to spare memory if possible
    n = Csize(1);
    o = Csize(2);
end

if (strcmpi(func2str(outop), 'sum') && strcmpi(func2str(inop), 'times')) % fallback to MatLab BLAS matrix multiplication if possible
    C = A*B;
else

    if (islogical(A) && issparse(A))
        warning('genmtimes.m: A is logical and sparse! But bsxfun wont work with sparse logical matrices. Automatically casting double(A).');
        A = double(A);
    end
    if (islogical(B) && issparse(B))
        warning('genmtimes.m: B is logical and sparse! But bsxfun wont work with sparse logical matrices. Automatically casting double(B).');
        B = double(B);
    end

    C = [];
    if issparse(A) || issparse(B)
        C = sparse(o,n);
    else
        C = zeros(o,n);
    end

    A = A';
    parfor i=1:n
        C(:,i) = outop(bsxfun(inop, A(:,i), B))';
    end
    C = C';
end

% C = [];
% if issparse(A) || issparse(B)
    % C = sparse(o, n);
% else
    % C = zeros(o,n);
% end
% A = A';
% if ~symmetry_flag
    % for i=1:n
        % C(:,i) = outop(bsxfun(inop, A(:,i), B))';
    % end
% else
    % C(:,1) = outop(bsxfun(inop, A(:,1), A))';
    % if n > 1
        % keyboard
        % for i=2:n
            % C(:,i) = [outop(bsxfun(inop, A(:,i), A(:, i:end)))'; sparse(i-1, 1)];
        % end
    % end
    % C = C + % recopy over
% end
% C = C';

end


    % -- Semi-vectorized version 3 - a lot faster
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
        % TRICKS3: we create a sparse matrix instead of assigning 1s directly into the matrix, since it's a sparse matrix, any addition of a non-zero entry is costly. Followed the advices from http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
%           network = network + sparse(repmat((j-1)*l+messages(:,j), 1, c-(j-1)), bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end)), 1, n, n);
            % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%   end

    % -- Semi-vectorized version 4 - fastest after the vectorized version, and is a bit more memory consuming
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   combinations_count = sum(1:c);
%   rows = zeros(mgen, combinations_count);
%   cols = zeros(mgen, combinations_count);
%   parfor j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
        % TRICKS3: we create a sparse matrix instead of assigning 1s directly into the matrix, since it's a sparse matrix, any addition of a non-zero entry is costly. Followed the advices from http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
        % TRICKS4: precache all indexes and then add the new links all at once after the loop. A bit more memory consuming, but so much faster!
%           idxend = sum(c:-1:(c-(j-1))); % combinatorial indexing (decreases with higher j)
%           idxstart = idxend - (c-j);
%           rows(:, idxstart:idxend) = repmat((j-1)*l+messages(:,j), 1, c-(j-1));
%           cols(:, idxstart:idxend) = bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end));
            % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%   end
    % At the end, add all new links at once
%   network = network + sparse(rows, cols, 1, n, n);
