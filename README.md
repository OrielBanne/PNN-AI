# PNN-AI


**PNN AI**

steps for preparing a new experiment

1 - receive the new experiments data<br>
    i - copy the data to experiments folder (create a folder like 'ExpX')
     scp -r mail:/home/orielban/experiments/Exp4 .
     the dot at the end of the instruction is critical (copy all)<br>
2 - check there are images and data looks good<br>
3 - create movies to check the data, use PNN/movieLWIR.py and PNN/LWIR TimeLapse.ipynb<br>
4 - find the positions and add to the experiments file to the dictionary plant positions
    use PNN/FindingTheCirclesExp 4.ipynb to create the tables for plant positions<br>
5 - decide what are the dates of the run to use<br>
    cutoff dates where plants are too small<br>
    cutoff dates where plants start overlapping<br>
6 - enter the correct days at experiments file dictionary experiments_info<br>
7 - add normalizations to experiments dictionary<br>
    use PNN/datasets/exp_mod_norms.py<br>
8 - set the labels correctly - add classes and labels to labels file<br>

steps for running an experiment:

1 - open parameters file in train folder - <br>
2 - define all the parameters you need<br>
3 - run train_loop_2<br>
4 - numeric results will be displayed to the screen<br>
5 - visueal results will be saved to Results folder inside PNN<br>
    make sure after run to copy all the results to a specific folder you name for the specific experiment<br>



