Benchmarking Datasets for Unsupervised Anomaly Detection
========================================================

These datasets can be used for benchmarking unsupervised anomaly detection 
algorithms (for example LOF).

The datasets have been obtained from multiple sources and are mainly based on 
datasets originally used for supervised machine learning. By publishing these 
modifications, a comparison of different algorithms is now possible for 
unsupervised anomaly detection.

The original datasets have some citation policy, which must be followed when 
used for research. Please refer to the list below.

Some remarks:
* the CSV file uses "," as separator and "." as a decimal point
* The label of each record is in the last column, indicating a normal record 
  "n" or an outlier "o".
* The data might be sorted (so consider shuffling if required)
* The data columns are not normalized (most algorithm require to normalize 
  data first)
* The data contains only real numbers
* In some cases, the scientific representation is used (e.g. 7.929E-4,0). When 
  parsing, please take care of correct handling. (Java/Python automatically 
  takes care about it)
* The label is only for accuracy computation, you may not use it for parameter 
  estimation (such as selecting a "good" k for LOF)

Citation Policies of Datasets
=============================

aloi
----
J. M. Geusebroek, G. J. Burghouts, and A. W. M. Smeulders, The Amsterdam 
library of object images, Int. J. Comput. Vision, 61(1), 103-112, January, 
2005. 

E. Schubert, R. Wojdanowski, A. Zimek, H.-P. Kriegel, On Evaluation of Outlier 
Rankings and Outlier Scores, In Proceedings of the 12th SIAM International 
Conference on Data Mining (SDM), Anaheim, CA, 2012. 

annthyroid
----------
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of 
Information and Computer Science. 

Schiffmann W, Joost M, Werner R. Synthesis and Performance Analysis of
Multilayer Neural Network Architectures. University of Koblenz; 1992.

breast-cancer
-------------
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science. 

Mangasarian OL, Street WN, Wolberg WH. Breast Cancer Diagnosis and
Prognosis via Linear Programming. SIAM News. 1990;23(5):1–18.

kdd99
-----
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science. 

letter
------
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science.

Micenkova B, McWilliams B, Assent I. Learning Outlier Ensembles: The Best of 
Both Worlds - Supervised and Unsupervised. In: Proceedings of the ACM ODD2. 
2014. p. 51–54.

P. W. Frey and D. J. Slate. "Letter Recognition Using Holland-style Adaptive 
Classifiers". (Machine Learning Vol 6 #2 March 91) 

pen-*
----------
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science.

F. Alimoglu (1996) Combining Multiple Classifiers for Pen-Based Handwritten 
Digit Recognition, MSc Thesis, Institute of Graduate Studies in Science and 
Engineering, Bogazici University.

satellite
---------
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science.

shuttle
-------
Lichman, M. (2013). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School 
of Information and Computer Science.

Thanks to NASA for allowing us to use the shuttle datasets. 

speech
------
Niko Brümmer, Sandro Cumani, Ondrej Glembek, Martin Karafiat, Pavel Matejka, Jan 
Pesan, Oldrich Plchot, Mehdi Soufifar, Edward de Villiers and Jan Cernocky.
Description and analysis of the Brno276 system for LRE2011. In Proc. of Odyssey
2012: The Speaker and Lang. Rec. Workshop, 2012.

Micenkova B, McWilliams B, Assent I. Learning Outlier Ensembles: The Best of 
Both Worlds - Supervised and Unsupervised. In: Proceedings of the ACM ODD2. 
2014. p. 51–54.

