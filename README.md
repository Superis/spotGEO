# spotGEO
Detecting orbit objects on sky telescope images

THE PROBLEM

The competitionâ€™s topic concerns the correct detection and localization of Geo orbit objects that exist in sky images taken from different telescopes.

Full desc here: https://kelvins.esa.int/spot-the-geo-satellites/problem/ 


THE DATA

Train-data consist of 1280 sequences of images. Every sequence contains 5 frames, which means that is made of 5 different consecutive captions of the same sky. 
The images are annotated per frame, in each frame there is information about the number of our target objects and their exact coordinates. Not every object has to be visible in every frame. In order to consider object as an existent, it has to be visible in at least 3 of the 5 sequence_frames.

Full desc here: https://kelvins.esa.int/spot-the-geo-satellites/data/ 

METRICS

The metrics used to evaluate a valid submission are based on a provided script. The ranking is computed based on F1 metric of predictions (5120 sequences) and MSE as tie-breaker.
For a frame prediction of objects and coordinates, a matching is performed between objects and predictions and we consider True Positives,False Positives and False Negatives.
There is an error threshold Îµ that if it's bigger than matching prediction distance, the prediction is a true positive and regression error 0 . If it is smaller than this but prediction smaller than Ï„, it's still true positive with regression error d(x,y). If it's bigger than Ï„, it's false positive with regression error t^2 and also object considered false negative with extra regression error of t^2. 

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

Pixel Classifier

Intelligent Sampling of Negatives

The choice of negative samples that are going to be included in the training set is being made through a random process in the starter-kit. So, it's reasonable to try a different approach and provide a classifier with pixels that are easily misclassified because of their similarity with the positive ones. An idea is to find the average luminance/value of all positive pixels and then choose the nearest negative samples to this value. Some different test/trials are executed to make the best representation possible for our negative dataset in comparison with the initial random choice.


Extra positive samples

The positive samples of the training set are very specific. The pixel coordinates are recorded in the annotation file, and we store the pixels' neighbourhood to construct the positive records of the training set. We can try to consider as positives all the pixels that are located around the initial positive pixel if the distance is smaller than a radius=r.



BASELINE STARTER KIT


255358.02544633413
0.9948781372052422

NEGATIVES=3 SVC RAD3
Recall 86.7 Precision 26.3
Compute score... 0.002768038383465584 0.043478260869565216
Score: 0.994795, (MSE: 7976.412073)


NEGATIVES==20 SVC RAD3
Recall 65.5 – Precision 41.4
Compute score... 0.010656333252603536 0.12753623188405797
Score: 0.980331, (MSE: 7918.724769)

NEGATIVES==20 SVC RAD2
Recall 66.8 – Precision 40.1
Compute score... 0.010306778222383897 0.12318840579710146
Score: 0.980978, (MSE: 7915.970767)

NEGATIVES==30 SVC RAD2
Recall 53.6 - Precision 46.2
Compute score... 0.013619201725997843 0.1463768115942029
Score: 0.975080, (MSE: 7891.244659)

NEGATIVES==20 RF RAD3 (n_estimators 20, depth 20)
Recall 75.9 Precision 36.7
Compute score... 0.0033577803304762747 0.05507246376811594
Score: 0.993670, (MSE: 7973.342276)

NEGATIVES==20 RF RAD3
Recall 82 -Precision 40.5
Compute score... 0.0038697498820198205 0.059420289855072465
Score: 0.992734, (MSE: 7962.415301)

NEGATIVES==30 SVC RAD2
FILTERING 2<=X<=3
Compute score... 0.026354801515401087 0.2318840579710145
Score: 0.952670, (MSE: 7818.976658)

NEG30,RAD2
FILTER 3<=X<=10
Compute score... TP 176 FP 5228 FN 514
0.03256846780162842 0.25507246376811593
Score: 0.942238, (MSE: 7765.724400)

C:\Users\Peris\Downloads>validation.py my_submissionSmall_nothread_rad2_neg5_svc_train1201-1280noRandfilter3-10.json train_anno.json
Validating... passed!
Compute score... TP 216 FP 6795 FN 474
0.03080872913992298 0.3130434782608696
Score: 0.943903, (MSE: 7778.473495)

C:\Users\Peris\Downloads>validation.py my_submissionSmall_nothread_rad2_neg5_svc_train1201-1280noRandfilter1.json train_anno.json
Validating... passed!
Compute score... TP 90 FP 10063 FN 600
0.00886437506155816 0.13043478260869565
Score: 0.983399, (MSE: 7922.595368)

C:\Users\Peris\Downloads>validation.py my_submissionSmall_nothread_rad2_neg30_svc_train1201-1280noRandfilter3-10.json train_anno.json
Validating... passed!
Compute score... TP 35 FP 2480 FN 655
0.013916500994035786 0.050724637681159424
Score: 0.978159, (MSE: 7616.556239)

?ntelligent sampling of negative pixels

Neg20-Rad2-Filter3-10
Precision 56.9– Recall 34.8
Compute score... TP 47 FP 3398 FN 643
0.013642960812772133 0.06811594202898551
Score: 0.977267, (MSE: 7722.822991)

Neg20-Rad2-Filter1-10
Compute score... TP 163 FP 8593 FN 527
0.018615806304248514 0.23623188405797102
Score: 0.965488, (MSE: 7829.223239)


python validation.py my_submissionSmall_nothread_rad2_neg20_svc_train1201-1280noRandintellfilter2-3.json train_anno.json
Validating... passed!
Compute score... TP 99 FP 3901 FN 591
0.02475 0.14347826086956522
Score: 0.957783, (MSE: 7834.228875)

SVM-RBF
my_submissionSmall_nothread_rad3_neg50_SvmRbf_train1201-1280filter2-3.json
recall: 83.7 precision: 54.9
Compute score... TP 26 FP 2419 FN 164
0.01063394683026585 0.1368421052631579
Score: 0.980266, (MSE: 1980.495822)




