gbnn-matlab
===========

Gripon-Berrou Neural Network implementation in Octave 3.8.1 and also working on Matlab 2013a.

See main.m for more informations and an example of usage, or simple.m or gbnn_mini.m if you want even simpler versions to kickstart you.

Although it is fully functional for all implemented features, this was a work-in-progress, intended for more features to be added or changed (but since the addition of named optional arguments, this shouldn't affect you and future changes should be mostly retro-compatibles), although development has now stopped as of 2020. Most features in the package were optimized for speed under the Matlab version at the time, and it should also be compatible with Octave although with less speed optimizations (some Octave functions with equivalents in Matlab had a rough non optimized implementation at the time, this may have changed since then).

If you want to use the GBNN network in your own application, you only need the files inside the `gbnn-core/` folder. The other folders and files are just usage examples (such as main.m) or figures generation scripts, some of which were used in the published conference paper:

Larroque, Stephen & Sedgh Gooya, Ehsan & Gripon, Vincent & Pastor, Dominique. (2015). Using Tags to Improve Diversity of Sparse Associative Memories. 10.13140/RG.2.1.3079.3762. [http://dx.doi.org/10.13140/RG.2.1.3079.3762](http://dx.doi.org/10.13140/RG.2.1.3079.3762)

If you want to make your own simple implementation, you can take a look at [gbnn-core/gbnn_mini.m](https://github.com/lrq3000/gbnn-matlab/blob/master/gbnn-core/gbnn_mini.m), which is a minimal implementation of the GBNN network under 150 LoC.

This implementation in MATLAB/Octave was developed by Stephen Larroque and Vincent Gripon. Tagged network extension by Ehsan Sedgh Gooya.

It is licensed under CRAPL or MIT Public License at your convenience.
