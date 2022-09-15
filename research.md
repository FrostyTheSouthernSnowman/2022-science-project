# Quotes

"The cost function reduces all the various good and bad aspects of a possibly complex system down to a single number, a scalar value, which allows candidate solutions to be ranked and compared" — Page 155, Neural Smithing: Supervised Learning in Feedforward Artificial Neural Networks, 1999

"It is important, therefore, that the function faithfully represent our design goals. If we choose a poor error function and obtain unsatisfactory results, the fault is ours for badly specifying the goal of the search." — Page 155, Neural Smithing: Supervised Learning in Feedforward Artificial Neural Networks, 1999

"The gradient points in the direction in which a function increases most rapidly." - Page 983, Calculus: Early Transcendentals by Howard Anton, Irl Bivens, Stephen Davis, 2005

"With a binary search, doubling the size of the list merely adds one more item to look at" - https://www.cs.mtsu.edu/~xyang/2170/binarySearch.html

# Resources

- https://towardsdatascience.com/linear-regression-with-python-and-numpy-25d0e1dd220d
- https://scikit-learn.org/stable/modules/linear_model.html
- https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
- https://www.khanacademy.org/computing/computer-science/algorithms/binary-search/a/binary-search
- https://research.cs.queensu.ca/home/cisc101spring/Spring2006/webnotes/search.html#BinarySearch
- https://www.math.uni-bielefeld.de/documenta/vol-ismp/40_lemarechal-claude.pdf
- https://www.tutorialspoint.com/how-to-find-minimum-element-in-an-array-using-binary-search-in-c-language

# Bibliography

- Haskell B. Curry, Quart. Appl. Math Volume 2, 1944, 258-261 (found in https://www.ams.org/journals/qam/1944-02-03/S0033-569X-1944-10667-3/)
- Howard Anton, Calculus: Early Transcendentals, Anton Textbook, Inc., 2005
- Leon Bottou, Stochastic Gradient Learning in Neural Networks, 8/24/2022, https://leon.bottou.org/publications/pdf/nimes-1991.pdf
- Frank Pfenning, Lecture 6: Binary Search, 8/25/22, https://www.cs.cmu.edu/~15122-archive/n17/lec/06-binsearch.pdf
- Claude Lemarechal, Cauchy and the Gradient Method, 8/25/22, https://www.math.uni-bielefeld.de/documenta/vol-ismp/40_lemarechal-claude.pdf

# Background Research (Outline)

- Background on Optimization Algorithms

  - Minimizing error
    - solving equation in one variable
    - AI applications

- Different Existing Optimization Algorithms
  - Gradient Descent
    - Explain Gradient
    - Explain Pros/Cons
  - Binary Search
    - How It Usually Works
    - How I Adapted It
    - Explain Pros/Cons

# Background Research

Optimization algorithms are a class of algorithms in computer science that can find the minima of functions. Optimizer algorithms are widely used in developing artificial intelligence (AI) algorithms, but also apply to other areas in computer science. In practice, optimization algorithms are often used to minimize the error rate (a sort of "how bad am I?" score) of a model/AI algorithm. By making this "How bad am I?" score lower, you make your model better. If the error rate is zero, that means that you have found a solution. The purpose of optimization algorithms is to find this minimum error rate.

Gradient Descent is one of the earliest and most used optimization algorithms. Some form of gradient descent can be found in nearly every major AI algorithm. OpenAI's GTP-3, the youtube algorithm, and even google maps all use gradient descent in some way. The way gradient descent works is by calculating a mathematical concept called the gradient. The gradient always points the in direction that a function is increasing the fastest. By traveling opposite the direction of the gradient, gradient descent actually travels in the direction of the minimum.

Binary search is a simple, elegant, and highly efficient algorithm that is traditionally used to find elements in a list or database. There is also a adapted version of binary descent that can be used as an optimization algorithm. The algorithm works in two steps. First, find a range that contains a minimum of the function. Once a range is located, a slightly modified version of binary search can be used to find the minimum. Binary search cuts the range that contains the minimum in half until it can find a solution. Binary search has the advantage of relying less on gradients which should hopefully minimize the effects of or even solve a problem that has haunted gradient descent for decades, the "vanishing gradient problem".
