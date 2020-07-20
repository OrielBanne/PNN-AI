# PNN-AI


PNN AI

steps for preparing a new experiment

1 - receive the new experiments data
    i - copy the data to experiments folder (create a folder like 'ExpX')
     scp -r orielban@violeta.cs.technion.ac.il:/home/orielban/experiments/Exp4 .
     the dot at the end of the instruction is critical (copy all)
2 - check there are images and data looks good
3 - create movies to check the data, use PNN/movieLWIR.py and PNN/LWIR TimeLapse.ipynb
4 - find the positions and add to the experiments file to the dictionary plant positions
    use PNN/FindingTheCirclesExp 4.ipynb to create the tables for plant positions
5 - decide what are the dates of the run to use
    cutoff dates where plants are too small
    cutoff dates where plants start overlapping
6 - enter the correct days at experiments file dictionary experiments_info
7 - add normalizations to experiments dictionary
    use PNN/datasets/exp_mod_norms.py
8 - set the labels correctly

steps for running an experiment:

1 - open parameters file in train folder - 
2 - define all the parameters you need
3 - run train_loop_2
4 - numeric results will be displayed to the screen
5 - visueal results will be saved to Results folder inside PNN
    make sure after run to copy all the results to a specific folder you name for the specific experiment



