# spotGEO
Detecting orbit objects on sky telescope images

THE PROBLEM

The competition’s topic concerns the correct detection and localization of Geo orbit objects that exist in sky images taken from different telescopes.

Full desc here: https://kelvins.esa.int/spot-the-geo-satellites/problem/ 


THE DATA

Train-data consist of 1280 sequences of images. Every sequence contains 5 frames, which means that is made of 5 different consecutive captions of the same sky. 
The images are annotated per frame, in each frame there is information about the number of our target objects and their exact coordinates. Not every object has to be visible in every frame. In order to consider object as an existent, it has to be visible in at least 3 of the 5 sequence_frames.

Full desc here: https://kelvins.esa.int/spot-the-geo-satellites/data/ 

METRICS

The metrics used to evaluate a valid submission are based on a provided script. The ranking is computed based on F1 metric of predictions (5120 sequences) and MSE as tie-breaker.
For a frame prediction of objects and coordinates, a matching is performed between objects and predictions and we consider True Positives,False Positives and False Negatives.
There is an error threshold ε that if it's bigger than matching prediction distance, the prediction is a true positive and regression error 0 . If it is smaller than this but prediction smaller than τ, it's still true positive with regression error d(x,y). If it's bigger than τ, it's false positive with regression error t^2 and also object considered false negative with extra regression error of t^2. 

Full desc here: https://kelvins.esa.int/spot-the-geo-satellites/scoring/

STARTER-KIT

A starter-kit is provided to start with and experiment on the problem analysis. The approach is based on the following :
Every image consists of 640x480 pixels. We build a classifier that is pixel-independent. Specicifally, in each image we choose the pixels that are marked as coordinates in the annotation file, and mark them as positives. We also choose a number of random other pixels that are marked as negative.
Training features of the classifiers are constructed by extracting a number of m (n*n = window size, where n is based on radius from the center) pixel values around the pixel we examine.
So, if for example rad==3, the window size is (3+1+3)x(3+1+3) = 49 values (columns). 
In the starter-kit random forest is the chosen processing algorithm.
The results show that the predictions are much more than the actual object centers, so there is an extraction of objects that come of the predictions and the consecutive pixel regions detected.
After the object detections that appear in these regions, the next step is filter the objects according to some criteria(e.g size) and the remaining ones' centers are printed on the submission files.
Starting pack submits a score of 0.995 as 1-F1, which means that performs poorly on the final results.
However, accuracy before the filtering in the pixel classifier is 0.99 (recall = 0.68, precision = 0.31 on the minority class that detects positive pixels).
We can infer that in this approach format, our sumbission depends on 2 different phases:
a. The pixel classifier that gives results on the pixel prediction.
b. The way that we process these results to have the best possible matching with the real objects.

So in first, we will try to improve seperately these two actions.


Comments for instructors
In folder results there are 2 csv files with results for different windows size, some different classifiers and different sampling of negative pixels.
The score of starter-kit has been recorder as 0.995 and the best until now as far as our trials are concerned is 0.98(so it's a bit improved) with LinearSVC,radius=3 and 30 negative samples per image which means a more imbalanced set, but achieves higher precision. 
