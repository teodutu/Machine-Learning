# Exercise 6 - SVM
For the first data set, a *linear kernel* is enough, but the second set requires the use of *gaussian kernels*. Similarly, the *SVM* for the third data set is trained using *gaussian kernels* by minimising the
cost of the classifier with regards to the parameters `C` and `sigma` on the *cross-validation* set.

The aim of the exercise is to filter spam email by using an *SVM*. Thus, the feature vector is composed
of those words selected form a *vocabulary* of frequently used spam words which appear in the current
email. The *SVM* is particularly accurate for this kind of task: `98.5%`.
