# Exercise 2 - Logistic Regression and Forward Propagation in Neural Networks
- the first part is nearly identical to the previous exercise, but now there are 10 classes;
- we predict each class individually and for the final prediction we pick the one with
the highest probability out of the 10 choices;
- the same problem is then implemented using a given neural network, by applying *feedforward
propagation**.

## One Versus All Prediction
- the cost function `J` and its gradient `grad` are calculated identically to the way they
were for the last exercise;
- now, we need 10 predictions in order to classify the test cases;
- in order to achieve this in an optimised fashion, the *Octave* function `fmincg` is used;
- the prediction itself selects the class with the highest probability for each input.

## Neural Network: Forward propagation
- we calculate each *activation* (`a2` and `a3`), also taking care t add a line of *ones*
as the first line of `a2`;
- then, the exact predictions are extracted from `a3` in exactly the same manner as in
the *One Versus All* algorithm.
