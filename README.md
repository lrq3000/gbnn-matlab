gbnn-matlab
===========

Gripon-Berrou Neural Network implementation in Octave 3.8.1 and also working on Matlab 2013a.

See main.m for more informations and an example of usage, or simple.m or gbnn_mini.m if you want even simpler versions to kickstart you.

This is a work-in-progress, a lot of features may be added or changed (but since the addition of named optional arguments, this shouldn't affect you and future changes should be mostly retro-compatibles).

If you want to use the GBNN network in your own application, you only need the files inside the `gbnn-core/` folder. The other folders and files are just usage examples or figures.

If you want to make your own simple implementation, you can take a look at [gbnn-core/gbnn_mini.m](https://github.com/lrq3000/gbnn-matlab/blob/master/gbnn-core/gbnn_mini.m), which is a minimal implementation of the GBNN network under 150 LoC.

This implementation in MATLAB/Octave was developed by Stephen Larroque and Vincent Gripon. Tagged network extension by Ehsan Sedgh Gooya.

It is licensed under CRAPL or MIT Public License at your convenience.
