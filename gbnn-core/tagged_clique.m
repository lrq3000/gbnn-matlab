function [err1, err2, err3, timing, W_weighted_autoassociative, messages, erase] = tagged_clique(c, l, m, iterations)
% Tagged networks
% Main function, it sets the global variables and calls the other functions - FIXED
% AUTHOR: EHSAN SEDGH GOOYA
% Comments by STEPHEN LARROQUE, thus there may be some errors since I'm not the author of this code.
% This works on a Behrooz network without sparsity (Chi = c)
    fprintf('Tagged cliques: working on messages batch (%i)\n', m); flushout();
    messages = create_msg(c, l, m) ;
    erase = erasure(c, m) ;
    [err1 , err2 , err3 , timing, W_weighted_autoassociative] = weighted_autoassociative(c, l, m, messages, erase, iterations) ;
    %save('err_tagged_cliques.mat' , 'err_tagged_cliques') ;
    %save('timeT_tagged_cliques.mat' , 'timeT_tagged_cliques') ;
end

function messages = create_msg(c, l, m)
% Create a matrix of messages (not thrifty, they will be converted into thrifty on the fly when pushed into the network)
   messages = randi(l , [m , c]) ;
   %save('messages.m' , 'messages') ;
end

function erase = erasure(c, m)
% Prepare the list of clusters to erase per message (by default, floor(half the clusters) will be erased)
   erase = zeros(m , floor(c/2)) ; % pre-allocate the matrix/array that will contain the clusters indices to erase
   u = [] ;
   for i = 1 : m % for each message
       while length(u) ~= floor(c/2) % until we have enough erasures for this message
             u = [u randi(c)] ; % pick a random cluster index
             u = unique(u) ; % check that this cluster index wasn't already picked up
       end
       erase(i , :) = u ; % at the end, we have enough erasures for this messages, we append the erasures indices into our matrix erase
   end
   %save('erase.mat','erase') ;
end

function [err1 , err2 , err3 , time, W_weighted_autoassociative] = weighted_autoassociative(c, l, m, messages, erase, iterations)
% Main subfunction to learn/construct a tagged clique network and then decode partially and randomly erased messages
    W_weighted_autoassociative = zeros(c * l , c * l) ; % c*l = n % Pre-allocate the network (adjacency matrix)
    err1 = 0 ;
    err2 = 0 ;
    err3 = 0 ;
    time = 0 ;
    % Learn each message into the network
    fprintf('Learning...\n'); flushout();
    W_weighted_autoassociative = learn_weighted_autoassociative(W_weighted_autoassociative, c, l, messages) ; % learn this message
    % Correction phase
    fprintf('Correcting...\n'); flushout();
    t = cputime() ; % for perfs. NB: NEVER call cputime() or tic()/toc() inside a loop, this will slow things down a lot!!!
    for i = 1 : m % test for each message
        msg = messages(i , :) ; % select a message
        msgLorig = (0 : c - 1) * l + msg ; % shift offsets = conversion into thrifty code on-the-fly
        msgL = msgLorig ;
        msgL(erase(i , :)) = [] ; % erase randomly some clusters (by default half the clusters will be erased)
        for o = 1 : iterations
            [decod , idx] = decoding_weighted_autoassociative(W_weighted_autoassociative, msgL) ;
            msgL = decod ;
            if length(decod) >= c % if, with one iteration of decoding, we still don't have as many clusters as there should be, we proceed onto 4 more iterations of decoding to try to get the remaining (still unactivated) clusters.
                break ;
            end
        end
        % Error counting
        if idx ~= i % error1: is the recovered tag correct?
            err1 = err1 + 1 ;
        end
        if length(decod) ~= c % error2: have we recovered as many clusters as there are? (this does NOT check the content of the clusters!)
            err2 = err2 + 1 ;
        end
        if (length(decod) ~= c) || (any(sort(decod) ~= sort(msgLorig))) % error3: real error in decoding (if one bit doesn't match with the original message, then it's an error)
            err3 = err3 + 1 ;
        end
    end
    time = cputime() - t ; % for perfs

    % Compute the error rates (normalize over the total number of messages)
    err1 = err1 / m ;
    err2 = err2 / m ;
    err3 = err3 / m ;
end

function W_weighted_autoassociative = learn_weighted_autoassociative(W_weighted_autoassociative, c, l, messages)
% Learn a message into the network (adjacency matrix) with M tags!
    for i = 1 : size(messages, 1)
        %idx = randi(l) ; % UNUSED. idx should represent the tag (thus the message id), so why trying to set a random one based on l?
        msgL = (0 : c - 1) * l + messages(i , :) ; % pick up a message and shift offsets to prepare the message to be pushed into the network (this converts into thrifty codes on-the-fly)
       W_weighted_autoassociative(msgL , msgL) = i ; % Push the message to learn by index and assign the message id as the edges tags
   end
end

function [decoded_fanals , major_tag] = decoding_weighted_autoassociative(W_weighted_autoassociative, msgL)
% Decode a partially erased message, using a previously learnt tagged cliques network

    % Propagation Sum-Of-Sum
    g = sum(sign(W_weighted_autoassociative(msgL , :))) ; % Propagation: push the message and get the score for each fanal, this is the Sum-Of-Sum (equivalent to thrifty(msgL) * W_weighted_autoassociative)

    % Filtering GWTA
    decoded_fanals  = find(g == max(g)) ; % Filtering: Global Winner-Take-All (GWTA), we get the full message (with offsetted indices, but we can easily disoffset them)

    % Filter edges going outside the clique
    decoded_edges = W_weighted_autoassociative(decoded_fanals , decoded_fanals) ; % fetch tags of only the edges of the recovered clique (thus edges connected between fanals in the recovered clique but not those that are not part of the clique, ie connected to another fanal outside the clique, won't be included)

    % Filter useless fanals (fanals that do not possess as many edges as the maximum - meaning they're not part of the clique). Note: This is a pre-processing enhancement step, but it's not necessary if you use fastmode(nonzeros(decoded_edges)) (the nonzeros will take care of the false 0 tag)
    fanal_scores = sum(sign(decoded_edges));
    winning_score = max(fanal_scores);
    gwta_mask = (fanal_scores == winning_score);
    %if numel(decoded_edges(gwta_mask, :)) ~= numel(decoded_edges); keyboard; end;
    decoded_edges = decoded_edges(gwta_mask, :) ; % TODO: Not a bug, it indeed work, even if it should be decoded_edges(:, gwta_mask) instead of decoded_edges(gwta_mask, :). So I don't know why this works! But anyway this is optional, but it indeed enhances performances a bit with tagged networks (but this doesn't enhance performances on non-tagged networks).

    %decoded_fanals = decoded_fanals(gwta_mask);
    %major_tag = 1;

    % Filter edges having a tag different than the major tag, and then filter out fanals that gets disconnected from the clique (all their incoming edges were filtered because they were of a different tag than the major tag)
    major_tag = min(fastmode(nonzeros(decoded_edges))) ; % get the major tag (the one which globally appears the most often in this clique). NOTE: nonzeros somewhat slows down the processing BUT it's necessary to ensure that 0 is not chosen as the major tag (since it represents the absence of edge!).
    decoded_edges(decoded_edges~=major_tag) = 0 ; % shutdown edges who haven't got the maximum tag
    decoded_fanals = decoded_fanals(sum(decoded_edges)~=0) ; % kick out fanals which have no incoming edges after having deleted edges without major tag (ie: nodes that become isolated because their edges had different tags than the major tag will just be removed, because if these nodes become isolated it's because they obviously are part of another message, else they would have at least one edge with the correct tag).
end

function [answer] = isOctave
% Returns true if this code is being executed by Octave.
% Returns false if this code is being executed by MATLAB, or any other MATLAB variant.
    persistent octaveVersionIsBuiltIn; % cache result
    if (isempty(octaveVersionIsBuiltIn))
        octaveVersionIsBuiltIn = (exist('OCTAVE_VERSION', 'builtin') == 5); % exist returns 5 to indicate a built-in function.
    end
    answer = octaveVersionIsBuiltIn; % isOctave cannot be persistent since this is the return variable
end

function flushout
    if isOctave
        fflush(stdout);
        fflush(stderr);
    else
        drawnow('update');
    end
end

% Fastmode: mode implementation by Harold Bien that can return multiple results if multiple values are modes (while MatLab returns the smallest by default, and there's no way to change that).
function [y, n]=fastmode(x)
    % FASTMODE  Returns the most frequently occuring element
    % 
    %   y = fastmode(x) returns the element in the vector 'x' that occurs the
    %   most number of times. If more than one element occurs at equal
    %   frequency, all elements with equal frequency are returned.
    % 
    %   [y, n] = fastmode(x) does the same as above but also returns the
    %   frequency of the element(s) in 'n'.
    % 
    %   Note that due to speed considerations, no error checking is performed
    %   on the input data. Matrices will be reduced to vectors via the colon
    %   (:) operator, NaN's are ignored.
    % 
    %   Example
    %   % Generate a data set of values between 0 and 9
    %   >> data=fix(rand(1000,1).*9);
    %   % Get the most common value and its frequency
    %   >> [mode, n]=fastmode(data);
    %   % To confirm, run this simple script
    %   >> for i=1:9 disp(sprintf('Element %d: Frequency %d', i,
    %   length(data(data==i)))); end;
    % 
    %   % Note if you give it only unique values, all values will be
    %   % returned with a frequency count of 1, i.e.
    %   >> [y, n]=fastmode([1:9]);
    %   % will result in y=[1:9] and n=1
    % 
    %   See also MEAN, MEDIAN, STD.

    %   Copyright 2006 Harold Bien
    %   $Revision: 1.0 $    $Date: 2006/04/13 $

    % This code was evolved from mode.m by Michael Robbins and critical input
    % by others in MathWorks FileExchange

    % The data must be sorted in order for this algorithm to work
    sorted=sort(x(:));
    % Compute element-by-element difference. This will return 0 for
    % identical valued elements (since it is sorted) and non-zero for 
    % different elements. We add a dummy element at the end in order to
    % pick up repeated elements at the end (make sure last element is not
    % equal to the next-to-last element). This value will never be used for
    % mode computations, so it's safe to add in.
    dist=diff([sorted; sorted(end)-1]);
    % This gives us unique values (presumably faster than unique())
    % [Profiler indicates a lot of time spent here, so we avoid it]
    %unique_vals=sorted(dist~=0);
    % Find locations of all non-zero elements and take the difference
    % which counts how many of each element is there. The first repeated
    % elements are dropped in the dist, so we add it back in as the first
    % non-zero index, i.e. the first non-zero index indicates that many
    % repeated first entries
    % Get non-zero entries
    idx=find(dist);
    num=[idx(1); diff(idx)];
    % Get the mode, including possible duplicates
    n=max(num);
    % Pull the values from the original sorted vector
    y=sorted(idx(num==n));
end
