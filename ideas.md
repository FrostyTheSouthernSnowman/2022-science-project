# Can BSBP Find Minimums Faster Than Traditional Backprop/RL
## MDBS: MultiDimensional Binary Search.
### V1
MultiDimensional Binary Search is an algorithm for training Neural Networks.
First, you take two initial network configurations. One is ranked as better than the other.
Take the change between each of the parameters and use that as the gradient. 
Update the weights in the direction of the gradient times a learning rate. 
If the new model is better than the old one, repeat the process.
If not, the new model has overshot the right values. 
Take the old model and only update it by half the learning rate. If the new model is better, repeat the process.
If not, then compare the new model to the older overshot model.

m1 -> m2 if good, m2 -> m3
m1 -> m2 if bad, m1 -> half m2, if good, continue, else (half m2) -> m2

### V2
MultiDimensional Binary Search is an algorithm for training Neural Networks.
First, you take two initial network configurations. One is ranked as better than the other.
Take the change between each of the parameters and use that as the gradient. 
Update the weights in the direction of the gradient times a learning rate. 
If the new model is better than the old one, repeat the process.
If not, the new model has overshot the right values. The minimum must lie between the newest and third newest network.

m1 -> m2 if good, m2 -> m3
m1 -> m2 -> m3, m3 is worse, mid(m1, m2) = m4, mid(m2, m3) = m5, if m4 > m5, repeat with m2 -> m4, else repeat with m2 -> m5

## V3: A Slightly Different Idea Rooted in the Same Algorithm
Use the old protocol as a way of approximating gradients.