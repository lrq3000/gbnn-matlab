% #### Gripon-Berrou Neural Network, aka thrifty clique codes ####
% Original implementation in Octave by Vincent Gripon.
% Optimized (compatibility Octave/MatLab, vectorizations, binary sparse matrices, etc.), comments additions and functionality extensions by Stephen Larroque.
% Coded and tested on Octave 3.6.4 Windows and MatLab 2013a Linux. No external library nor toolbox is needed for this software to run.
% To run the code even faster, first use MatLab (which has a sparse-aware bsxfun() function) and use libraries replacements advised in https://research.microsoft.com/en-us/um/people/minka/software/matlab.html and also sparse2 instead of sparse as advised at http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
% The code is also parallelizable, just enable multithreading in your MatLab config. You can also enable the use of GPU for implicit GPU parallelization in MatLab and/or for explicit GPU parallelization (to enable the for loops or if you want to convert some of the code to explicitly process on GPU) you can use third-party libraries like GPUmat for free or ArrayFire for maximum speedup.
%
% ## Software architecture ##
% The software is composed of the following files:
%- gbnn_learn.m allows to learn a network and optionally generate a random set of messages (or you can provide your own). This will return the network and optionally the thrifty messages generated.
%- gbnn_test.m allows to test the correction capability of a previously learned network on randomly tampered messages (either by erasure or by noise). This will return an error rate.
%- gbnn_predict.m is used by gbnn_test.m and allows to correct a set of tampered messages (but the error rate won't be computed since this function will just try to recover the tamperings).
%- gbnn_messages2thrifty.m is used internally so that you can use full messages (eg: 1203) and internally it will be automatically converted into sparse thrifty messages (eg: 1000 0100 0000 0010).
%- gbnn_aux.m is a set of auxiliary functions.
%- gbnn_mini.m is a standalone, minified version of the script, with minimum comments and lines of codes (and features are reduced too). This file is intended as a quickstart to quickly grasp the main concept. You can first study this file, and when you feel confident, you can try analyzing the other gbnn files.
%- main.m is this file, an easy entry to use the gbnn algorithm.
%- simple.m is a simplified version of main.m, containing the bare minimum to use the gbnn network.
%- figX.m are example codes that show how to automatically generate a plot to compare the performances between whatever variables you want.
%
% ## Variables list ##
% -- Data variables
%- Xlearn : messages matrix to learn from. Set [] to use a randomly generated set of messages. Format: m*c (1 message per line and message length of c), and values should be in the range 1:l. This matrix will automatically be converted to a thrifty messages matrix.
%- Xtest : thrifty messages matrix to decode from (the messages should be complete without erasures, these will be made by the algorithm, so that we can evaluate performances test!). Set [] to use Xlearn. Format: attention this is a sparse matrix, it's not the same format as Xlearn, you have to convert every character into a thrifty code.
%
% -- Network/Learn variables
%- m : number of messages or a matrix of messages (composed of numbers ranging from 1 to l and of length/columns c per row).
%- miterator : messages iterator, allows for out-of-core computation, meaning that you can load more messages (greater m) at the expense of more CPU (because of the loop). Set miterator <= m, and the highest allowed by your memory without running out-of-memory. Set 0 to disable.
%- l : number of character neurons (= range of values allowed per character, eg: 256 for an image in 256 shades of grey per pixel). These neurons will form one cluster (eg: 256 neurons per cluster). NOTE: only changing the number of cluster or the number of messages can change the density since density d = 1 - ( 1 - 1/l^2 )^M
%- c : cliques order = number of nodes per clique = length of messages (eg: c = 3 means that each clique will at most have 3 nodes). If Chi <= c, Chi will be set equal to c, thus c will also define the number of clusters. NOTE: c can also be a vector [min-c max-c] to enable variable length messages.
% NOTE: increasing c or decreasing miterator increase CPU usage and runtime ; increasing l or m increase memory usage.
%- Chi (2014 update) : number of clusters, set Chi > c to enable sparse_cliques if you want c to define the length of messages and Chi the number of clusters (where Chi must be > c) to create sparse cliques (cliques that don't use all available clusters but just c clusters per one message).
%
% -- Test variables
%- network : specify the network to use that you previously learned.
%- thriftymessagestest : matrix of thrifty messages (only composed of 0's and 1's) that will be used as a test set.
%- erasures : number of characters that will be erased (or noised if tampering_type == "noise") from a message for test. NOTE: must be lower than c!
%- iterations : number of iterations to let the network converge
%- tampered_messages_per_test : number of tampered messages to generate and try per test (for maximum speed, set this to the maximum allowed by your memory and decrease the number tests)
%- tests : number of tests (number of batch of tampered messages to test)
% NOTE2: at test phase, iteration 1 will always be very very fast, and subsequent iterations (but mostly the second one) will be very slow in comparison. This is normal because after the first propagation, the number of activated nodes will be hugely bigger (because one node is connected to many others), thus producing this slowing effect.
% NOTE3: setting a high erasures number will slow down the test phase after first iteration, because it will exacerbate the number of potential candidates (ie: the maximum score will be lower since there is less activated nodes at the first propagation iteration, and thus many candidate nodes will attain the max score, and thus will be selected by WTA). This is an extension of the effect described in note2.
%
% -- 2014 update ("Storing Sparse Messages in Networks of Neural Cliques" by Behrooz, Berrou, Gripon, Jiang)
%- guiding_mask : guide the decoding by focusing only on the clusters where we know the characters may be (by cheating and using the initial message as a guide, but a less efficient guiding mask could be generated by other means). This is only useful if you enable sparse_cliques (Chi > c).
%- gamma_memory : memory effect: keep a bit of the previous nodes value. This helps when the original message is sure to contain the correct bits (ie: works with messages partially erased, but maybe not with noise)
%- threshold : activation threshold. Nodes having a score below this threshold will be deactivated (ie: the nodes won't produce a spike). Unlike the article, this here is just a scalar (a global threshold for all nodes), rather than a vector containing per-cluster thresholds.
%- propagation_rule : also called "dynamic rule", defines the type of propagation algorithm to use (how nodes scores will be computed at next iteration). "sum"=Sum-of-Sum ; "normalized"=Normalized-Sum-of-Sum ; "max"=Sum-of-Max % TODO: only 0 SoS is implemented right now!
%- filtering_rule : also called "activation rule", defines the type of filtering algorithm (how to select the nodes that should remain active), generally a Winner-take-all algorithm in one of the following (case insensitive): 'WTA'=Winner-take-all (keep per-cluster nodes with max activation score) ; 'kWTA'=k-Winners-take-all (keep exactly k best nodes per cluster) ; 'oGWTA'=One-Global-Winner-take-all (same as GWTA but only one node per message can stay activated, not all nodes having the max score) ; 'GWTA'=Global Winner-take-all (keep nodes that have the max score in the whole network) ; 'GkWTA'=Global k-Winner-take-all (keep nodes that have a score equal to one of k best scores) ; 'WsTA'=WinnerS-take-all (per cluster, select the kth best score, and accept all nodes with a score greater or equal to this score - similar to kWTA but all nodes with the kth score are accepted, not only a fixed number k of nodes) ; 'GWsTA'=Global WinnerS-take-all (same as WsTA but per the whole message) ; 'GLKO'=Global Loser-kicked-out (kick nodes that have the worst score globally) ; 'GkLKO'=Global k-Losers-kicked-out (kick all nodes that have a score equal to one of the k worst scores globally) ; 'LKO'=Losers-Kicked-Out (locally, kick k nodes with worst score per-cluster) ; 'kLKO'=k-Losers-kicked-out (kick k nodes with worst score per-cluster) ; 'CGkWTA'=Concurrent Global k-Winners-Take-All (kGWTA + trimming out all scores that are below the k-th max score) ; 'oLKO'=Optimal-Loser-Kicked-Out (kick only exactly one loser per cluster and per iteration, and only if it's not the max score) ; 'oGLKO'=Optimal-Global-Loser-Kicked-Out (kick exactly one loser per iteration for the whole message, and only if not the max score, equivalent to GkLKO but with k = 1)
%- tampering_type : type of message tampering in the tests. "erase"=erasures (random activated bits are switched off) ; "noise"=noise (random bits are flipped, so a 0 becomes 1 and a 1 becomes 0).
%
% -- Custom extensions
%- residual_memory : residual memory: previously activated nodes lingers a bit and participate in the next iteration
%- concurrent_cliques : allow to decode multiple messages concurrently (can specify the number here)
%- silent : silence all outputs
%- debug : clean up some more memory if disabled
%
%
% ## Thrifty clique codes algorithm ##
% (quite simple, it's a kind of associative memory)
%
% == LEARNING
% - Create random messages (matrix where each line is a messages composed of numbers between 1 and l, thus l is the range of values like the intensity of a pixel in an image)
% - Convert messages to sparse thrifty messages (where each number is converted to a thrifty code, eg: 3 -> [0 0 1 0] if c = 4; 2 -> [0 1 0 0 0] if c = 5)
% - Learn the network by using a simple Hebbian rule: we link together all nodes/numbers of a message, thus creating a link (here we just create the "thrifty" adjacency matrix, thrifty because we encode links relative to the thrifty messages, not the original messages, so that later we can easily push and propagate thrifty messages. So the structure of the thrifty adjacency matrix is similar to the structure of thrifty messages)
%
% == PREDICTION
% - Generate tampered messages with random erasures of randomly selected characters
% - Try to recover the original messages using the network. In a loop, do:
%    * Update network state (global decoding): Push and propagate tampered messages into the network (a simple matrix product like in Hopfield's network)
%    * Keep winners, most likely characters (local thrifty code): Use winner-takes-all on each cluster: only winners (having max score) must remain active on each cluster. This is equivalent to say that for each character position in the message, we keep the character that has max score (eg: [2 1 0 1] with c = 2 and l = 2, the winners in the thrifty message will be [1 0 0 1] which can be converted to the full message [1 2])
%    * Compute error score: if there is at least one wrong character, error = error + 1
%
%
% ## Troubleshooting ##
%
% - At Learning messages into the network, you encounter the following error: memory exhausted
%   This means that your software hasn't enough contiguous memory to run the program.
%    If you use Octave (as of 2014, v3.6.4), you cannot do anything about it. The first reason is that Octave does not possess a sparse bsxfun yet, thus everytime bsxfun is used, all input datatypes are converted to full, and it also returns a full datatype, which is why you run out of memory. Also, there is a hard limit for memory which is 2GB, and there's nothing you can do about it. In this case, the only thing you can do is to lower the m, l and c variables.
%    If you use MatLab, you can increase your virtual memory file (swap), MatLab is not limited and can use any contiguous memory available. Also, its bsxfun implementation is totally sparse compatible, thus the code should be a lot faster and memory efficient than with Octave (but this may change in the future).
%
% - At Converting to sparse, thrifty messages, you encounter the error: ind2sub: subscript indices must be either positive integers or logicals
%   This means that the matrix you try to create is simply too big, the indices overflow the limit (even if there's no content!). If you use Octave (as of 2014), there's nothing you can do, matrices are limited by the number of indices they can have. If you use MatLab, there should be no such limitations.
%
% - I have some troubles understanding the code
%   The code has been commented as much as possible, but since it's as much vectorized as was possible by the authors, it's understandably a bit cryptic in some parts. If you have some troubles understanding what some parts of the code do, you can try to place the "keyboard" command just before the line that upsets you, then run the code. This will stop the code at runtime right where you placed the keyboard command, and let you print the various variables so that you can get a clearer idea of what's going on. Also feel free to decompose some complex inlined commands, by typing them yourself step by step to monitor what's going on exactly and get a better grasp of the data structures manipulations involved.
%   Also you should try with a small network and test set (low values between 2 and 6 for m, l, c and tampered_messages_per_test), and try to understand first how the thriftymessages and network are built since they are the core of this algorithm.
%   Also you can place a keyboard command at the end of the script, call one run and then use full(reshape([init ; out ; propag], n, [])) to show the evolution of the decoding process.
%   Finally, if you use keyboard, don't forget to "clear all" everytime before doing a run, else MatLab/Octave may cache the function and may not see your latest changes.
%
% - When I use Global (k-)Loser-Kicked-Out, my error rate is very high!
%   First check your gamma_memory value: if it's enabled, try to set it to 0. Indeed, what this does is that the network always remember the first input (partially erased message), so that the already known characters are always largely winning over recovered characters, and are thus kicked out by this type of filtering rule. If you disable the memory, this should avoid this kind of effect.
%
% - I tried the code but I get a weird error: concatenation operator not implemented for '<unknown type>' on sparse matrices
% This error happens on Octave when you do a bsxfun with a binary operator (eg: bsxfun(@and, ...) on two sparse matrices with different datatypes (eg: one is logical and the other one is not). In this case, just convert both matrices with logical().
% On Matlab a similar error may happen if you do a matrix multiplication between two logical matrices: in this case, just convert both of your matrices to double or uint.
%
% ## ToDo ##
% - Implement sparse_cliques (Chi) in gbnn_mini
% - variable_length complete implementation and tests (when c is a vector of two values)
% - propagation_rule 1 and 2 (normalized and Sum of Max)
% - convergence stop criterions: NoChange (no change in out messages between two iterations, just like Perceptron) and Clique (all nodes have the same value, won't work with concurrent_cliques)
% - place parfor loop back inplace of the main for loop inside gbnn_test.m when the codebase will be mature enough.
%

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 10E4;
miterator = 0;
c = 8;
l = 64;
Chi = 100;
erasures = floor(c/2);
iterations = 4;
tampered_messages_per_test = 200;
tests = 1;

enable_guiding = true;
gamma_memory = 0;
threshold = 0;
propagation_rule = 'sum'; % TODO: not implemented yet, please always set 0 here
filtering_rule = 'GWsTA';
tampering_type = 'erase';

residual_memory = 0;
concurrent_cliques = 2;
no_concurrent_overlap = true;
concurrent_successive = false;
GWTA_first_iteration = false;
GWTA_last_iteration = false;

silent = false; % If you don't want to see the progress output

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'miterator', miterator, 'l', l, 'c', c, 'Chi', Chi, 'silent', silent);
error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'l', l, 'c', c, 'Chi', Chi, ...
                                                                                  'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                  'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, 'tampering_type', tampering_type, ...
                                                                                  'residual_memory', residual_memory, 'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_successive', concurrent_successive, 'GWTA_first_iteration', GWTA_first_iteration, 'GWTA_last_iteration', GWTA_last_iteration, ...
                                                                                  'silent', silent);
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
