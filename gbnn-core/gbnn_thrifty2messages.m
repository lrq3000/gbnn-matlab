function messages = gbnn_thrifty2messages(thriftymessages, l)
% messages = gbnn_thrifty2messages(thriftymessages, l, miterator)
% Converts a sparse thrifty messages matrix into a dense messages matrix (with integers instead of binary)
% Messages matrix should be of size [m, n] (m lines of messages of length Chi*l each).
%

    %if ~exist('miterator', 'var')
    %    miterator = 0;
    %end

    [m, n] = size(thriftymessages);
    Chi = n/l;

    messages = reshape(thriftymessages', l, []);
    characters = mod(find(messages)-1, l)+1;
    messages = double(any(messages));
    if nnz(messages) < numel(characters)
        error('cannot convert concurrent thriftymessages (ie, with more than one fanal activated per cluster) into dense messages! If your message is not concurrent, check the orientation (try to transpose).');
    end
    messages(messages>0) = characters;
    messages = reshape(messages, Chi, [])';

    % Alternative way of doing the same thing
    %messages = double(reshape(thriftymessages', l, []));
    %messages(messages > 0) = find(messages);
    %messages(messages > 0) = mod(messages(messages > 0)-1, l)+1;
    %messages = reshape(max(messages), Chi, [])';

end