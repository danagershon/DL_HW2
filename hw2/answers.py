r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. A. The Jacobian tensor $\frac{\partial Y}{\partial X}$ is a 4D tensor of the shape: (64, 512, 64, 1024) since the 
shape of $Y$ is (64, 512) and the shape of $X$ is (64, 1024).

1. B. Yes it is sparse. Since each output row depends only on the corresponding input row (sample),
we get that only the elements: $\frac{\partial y_{i,j}}{\partial{x}_{i,k}}$ are non-zero.

1. C. No, we do not need to materialize the above Jacobian. Using the chain rule:
$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} * \frac{\partial Y}{\partial X} = 
\frac{\partial L}{\partial Y} * W^T$. So we only need to multiply by $W^T$.

2. A. The Jacobian tensor $\frac{\partial Y}{\partial W}$ is a 4D tensor of the shape: (64, 512, 512, 1024) since the 
shape of $Y$ is (64, 512) and the shape of $W$ is (512, 1024).

2. B. Yes it is sparse. Since each output element depends only on the weights that multiply the corresponding input
elements, we get that only the elements: $\frac{\partial y_{i,j}}{\partial{w}_{j,k}}$ are non-zero.

2. C. No, we do not need to materialize the above Jacobian. Using the chain rule:
$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} * \frac{\partial Y}{\partial W} = 
X^T * \frac{\partial L}{\partial Y}$. So we only need to multiply by $X^T$.

"""

part1_q2 = r"""
**Your answer:**

No. Backpropagation is not required for training neural networks with gradient-based optimization.
The gradients can be computed for example with forward AD or by hand, and there are alternatives like genetic algorithms 
that do not rely on gradients. 
However, backpropagation efficiently calculates gradients, significantly accelerating the training process.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.04, 0.08
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======

    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg = (
        0.1,
        0.03,  # vanilla SGD
        0.003,  # SGD with momentum
        0.0002,  # RMSProp
        0.0001,
    )
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======

    wstd, lr = (
        0.15,  # Weight standard deviation
        0.001,  # Learning rate
    )
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Without dropout we see high training accuracy but low test accuracy, because it easier for the model to overfit to the training dataset. With dropout=0.4 we see lower training accuracy, because connections are dropped which makes it harder to train, but the tradeoff is a higher test accuracy (only after epoch 15) which implies the model is successfully generalizing because of the dropout which is as expected.
We can also see this from the loss: The train loss for no dropout is rapidly dropping, even when the test loss starts going up - this implies overfitting rather than learning and generalizing.
With dropout we see slower converges in the train loss but the test loss improves over time.

2. We can see that high dropout (0.8) can be worse than low dropout (0.4) even in the test accuracy. We can explain this by considering that the learning is now slower because of the dropped connections, and the model can't establish strong connections and patterns that it could have in low dropout. We should note however that it does perform better than no dropout at all. Thus the best model overall is a low-dropout which allows the model to learn but also regularize and generalize to the test set without overfitting.

"""

part2_q2 = r"""
**Your answer:**
Yes.
Cross Entropy estimates the how much the model is confident in its answer (i.e the loss is lowest when the scores tend to 1 for the correct label, rather than being spread out). However, accuracy tests are accurate the model is after converting the score to class via softmax. 
Consider the following example for a binary classifier tested on two data samples:
Before training it gives:
1. For the first sample a high score to the correct label (i.e label=0, and scores=[0.9, 0.1])
2. For the second sample a medium score and misclassifies the label (i.e label=0, and scores=[0.45,0.55])

Then after training the model could improve to reach
Test #1 scores: [0.6,0.4], Test #2 scores: [0.55, 0.45]

And now accuracy has increased (50% to 100%) and the loss has increased (the scores are closer to 0.5, instead of tending to 1).

"""

part2_q3 = r"""
**Your answer:**
1. Back-propagation is the efficient calculation of the loss gradients with respect to every layer's parameters, by propagating the gradients via the chain rule through all the layers. GD however is the descent-based algorithm that actually improves the parameters using the gradients calculated. So in general back-prop calculates gradients but doesn't change anything, GD uses these gradients and changes the params.

2. SGD is the case of GD where the batch size is 1, so we calculate the gradient with respect to a single data point every step. In general GD calculates the gradient and averages it over a bigger batch. This difference means that the gradient steps the GD takes are smooth and less noisy but take longer in time and memory to calculate the gradient over many points just to move a little. Overall GD takes a lot of little noisy steps, sometimes improving and sometimes not but in Expectation after a long time it should converge similarly to SGD which takes long smooth and direct steps towards the minimum.

3. In Deep Learning the models are huge and have many parameters, so SGD is preferred to use instead of GD because the machine won't necessarily have enough memory to contain all the gradients for a big batch. And because each step is faster and takes less memory we can trade-off by just running SGD for a lot more steps and after a while we should approach the minimum.

4. A. Yes this method is equivalent. When you forward, the loss tensor will have its grads saved with it. So when we sum the losses together we sum the grads of all the batches. Later the backwards will use this sum of grads to backpropagate which is equivalent to summing the grads of all the batches and thus this method is equivalent to performing GD on all the batches at once.

B. As mentioned before, you have still have to save the grads of each loss tensor to sum them later. This means that even if the batches fit in the memory, the amounts of grads saved will be like in GD and we will reach an out of memory error.
"""

part2_q4 = r"""
**Your answer:**
1.
A. Instead of saving the grad at each step ($O(n)$), we can save just the last_grad & last_val to calculate the next ones in the chain (because the tree is serial). This way we reach O(1) memory complexity at the same time complexity.  
B. First we'll run a forward pass. In this pass we won't remember every value calculated but we'll do it slightly differently: Every k nodes in the tree we'll save a "checkpoint" - the value of the function in that node.
This way when we calculate the gradients of the last k nodes we first calculate the value of them by using the last checkpoint (and forward passing it) - this will take O(k) time and memory. Now we do normal backprop, and when we reach the next k-block we continue the same way but we forget the buffer of the k values saved from the last block. I.e we remember O(k) values and gradients at each block, and we still only calculate the values at $O(k \cdot n/k)=O(n)$ time. This reduces the memory to $O(n/k + k)$ [n/k checkpoints, k values in buffer] while maintaining $O(n)$ time. By choosing $k=O(\sqrt{n})$ we can reach $O(\sqrt{n})$ memory.
  
C. Yes, by doing the same methods for all paths from the input to the output (which is way less than the size of the tree) we can generalize it to any tree structure.

D. Very deep neural nets take a lot of memory and to train with GD we'll need even more memory to save all the gradients for all the params. This means that sometimes it is impossible to train a model in the given memory requirement without using this methods to reduce the memory complexity of back propagation. 

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. This is when the optimizer has a high error i.e, the optimizer is unable to push the model to the best local minima. This is possible to fix by tuning the hyperparameters for learning (i.e lr, reg, momentum) or to use a completely different Optimizer.

2. This is when the model has learned well on the training set but fails to "Generalize" good results to the test set. This is also known as overfitting and can be fixed by good regularization techniques like Dropout, BatchNormalization etc.

3. This is when the model simply fails to learn even from the training set (i.e Underfitting). This can be improved by making the model deeper (more complexity) which adds expressivitiy and approximation capabilities to the model. For example increasing the receptive field of a CNN, adding more layers to a MLP, and so on.
"""

part3_q2 = r"""
**Your answer:**

For example when training a binary model to classify if an image is the digit 0 or not, but in training we over-expose it to the digit 0. The model will learn that it's worth saying "True" most of the time to be correct, and will be punished less for being mistaken by guessing "True". So in deployment when it will see different digits most of the time it will guess "True" which leads to a high False Positive Rate.

The opposite case of high False Negative Rate can happen when we under expose it to the digit 0, i.e give it a lot of other digits and it will train to say "False" most of the time, and will miss a lot of zeros ("False Negative").
"""

part3_q3 = r"""
**Your answer:**
1. In this scenario we don't mind having a lot of false negatives (i.e missing people with the disease) because in that case they'll still survive and even show symptoms to allow for treatment later. So in this case we want a high True positive rate at a cost of a high False negative rate -> The optimal point of us is shifted to the top right of the ROC diagram.

2. In this scenario, even missed patient will result in death, and it is better to send them to the expensive test to even have a chance to save them. Thus we can't allow missing patients, and we need a low False Negative rate. This will come at the cost of a low True Negative rate -> The optimal point is shifted to the bottom left of the ROC diagram.

"""


part3_q4 = r"""
**Your answer:**
There are many disadvantages to using MLP for this task:
1. Sentence length: The MLP has a fixed size, which limits the size of the sentence the model allows. And also, in reality most of the sentences will be short and the MLP won't know how to train the weights that correlate to the last words that rarely appear in this fixed sentence size. This way when it gets a longer than average sentence, it will most probably not perform well.

2. Word Shifting: The MLP will have to relearn the sentiment for the sentences I like apples and Hi, I like apples, because every word is now in a different input node and the weights are not shared. This both means that the model has to learn multiple times for similar sentences, and that this will greatly hurt the model's performance because of similar but contradictory sentences like I like apples, I don't like apples.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


1. Number of params for a convolutional layer is $K\cdot (C_{in}\cdot F^2+1)$  

- For the CNN we have two 3x3 convs on $K=256, C_{in}=256$ which gives $2\cdot 256\cdot (256\cdot 3^2+1)=1,180,160$ params.  

- For the Bottleneck block we have:  
Layer 1: $F=1, K=64, C_{in}=256 \Rightarrow No.params = 64\cdot (256\cdot 1^2+1) = 16,448$  
Layer 2: $F=3, K=64, C_{in}=64 \Rightarrow No.params = 64\cdot (64\cdot 3^2+1) = 36,928$  
Layer 3: $F=1, K=256, C_{in}=64 \Rightarrow No.params = 256\cdot (64\cdot 1^2+1) = 16,640$  
Overall the Bottleneck has $70,016$ params which is way less than the naive CNN.

2. Number of operations for a convolutional layer is about $C_{in} \cdot I \cdot F^2 \cdot K$  
Where $I=W\cdot H$ - the spatial input_size for a layer, $O=W_{out}\cdot H_{out}$ - output_size for the layer.  
After forward pass, the size changes by a factor of about $O = I \cdot K/C_{in}$, 
Which is the conversion from $C_{in}$ channels to $K$ channels [Assuming padding is same - the spatial size in reality won't change the factor by much]

The amount of operations depends on the image input so we'll label it by $I=W\cdot H$.

For the CNN:
input_size doesnt change. Number of operations:  
- Layer 1: $256 \cdot I \cdot 9 \cdot 256$  
- Layer 2: $256 \cdot I \cdot 9 \cdot 256$  
Sum: $1,179,648 \cdot I$  

Bottleneck Operations:
input_size changes because the channels change. Number of operations: 
- Layer 1: input_size=$I$, number of operations: $256 \cdot I \cdot 1 \cdot 64$  
- Layer 2: input_size=$I/4$, number of operations: $64 \cdot (I/4) \cdot 9 \cdot 64$  
- Layer 3: input_size=$I/4$, number of operations: $64 \cdot (I/4) \cdot 1 \cdot 256$  
Sum: $29,696 \cdot I$  

So for $I=32^2$ we get:  
CNN: $1,207,959,552$ operations  
Bottleneck: $30,408,704$ operations  

The bottleneck has much less operations.

3. 
- (1) The CNN has better expressiveness spatially because it uses F=3 for all layers (which the bottleneck only uses F=3 for the middle layer) and the receptive field of each neuron effectively triples after each layer, which allows for deeper spatial combinations.
- (2) The bottleneck has more abilities to combine the input across feature maps because it has a bigger number of channels and can create deeper combinations from them, even if every featuremap is spatially simpler than the CNN.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
1. We can see that the correlation between depth and accuracy is not straight forward. But overall, the model with L=2 (very low depth) was easier to train and reached the highest training accuracy, but at the price of staying at a constant test_acc instead of improving which implies overfitting and the early stopping was necessary to prevent it from overfitting. Later L=4 provided a better tradeoff, giving less train-accuracy but higher overall test-accuracy which means it was better to generalize than L=2. However, further increasing the depth simply hurts the network: L=8 takes a lot longer to approach the accuracy by L=4 and it stops improving the test accuracy pretty early. And in the case of L=16 the model was completely untrainable and didn't learn anything.
This implies that too-deep models are harder to learn possibly because of the vanishing gradients problem and the detachment of the early layers from the result. Overall it seems medium sized depths are optimal, because less layers are easier to overfit but converge pretty quickly, while too deep models are slow and to train and we can barely reach the optimal weight configuration via GD like this.

2. Yes, L=16 was untrainable, because of it being too deep making GD highly ineffective (vanishing gradients problem).
To solve this we can either add skip connections between the early layers to deeper layers which will provide a more concrete gradient to work with, and also help converge faster (Residual Blocks). Another partial solution is to normalize the gradients and try to take bigger steps when calculating for the earlier layers, for the first few epochs, and then slowly lowering the learning rate to fine tune the last layers. This allows the model to first learn the earlier layers to put it in a good environment for a local minimum, and then tune the deeper layers to reach that minimum.

"""

part5_q2 = r"""
**Your answer:**
In this experiment we see that when L is fixed, we usually see that K=32 under performs in both test and accuracy compared to K=64,128.
The optimal amount of filters depends on L:
a. When L=2,4 we see that K=64 dominates both in train and test accuracy. This could be explained by considering that a higher K could lead to overfitting to specific filters and generally trying to find more patterns than necessary can hurt the training of the network and not generalize well in the test.
b. When L=8 we see that K=128 dominates both train and test accuracy. This could imply that in deeper models, increasing the amount of filters could help find stronger patterns in the deeper layers.

Overall it seems that this experiment shows that on shallow layers, the last layers can't form complex patterns, and increasing the number of filters doesn't improve the quality of the final patterns, just their amount.
While deeper models actually improve greatly when increasing the amount of filters because they can form complex patterns in the deeper layers, and greatly benefit from a larger amount of high quality patterns.

Compared to exp1_1 we see that we either want a medium depth model to learn quickly and generalize, or a deep model with a lot of filters for better quality results.

"""

part5_q3 = r"""
**Your answer:**

We can see that L=2 learns quickly and surpasses everyone in train accuracy, while quickly plateus in terms of test accuracy.
Compared to that, L=3 learns more slowly and doesn't reach a higher training accuracy but does eventually beat L=2 in terms of test accuracy. Lastly L=4 is the slowest of them all, and doesn't reach a high training accuracy nor a high test accuracy.

The L=4 case is explained because the model is so deep, it doesn't utilize the expressiveness of increasing the number of filters, but essentially learns "on two different blocks", plus the fact that the model is deeper and suffers from vanishing gradients, we understand why it takes longer and doesn't reach a high accuracy.

This experiment implies that L=3 (medium depth network) works better than the shallower network, even if they reach the same training accuracy. This is could be explained by considering that a shallow network doesn't have the depth to recognize small and meaningful patterns in the K=64 block, to be later passed to the K=128 block - this means that the patterns formed in the K=64 block are less meaningful and useful for the K=128 than they are in L=3 and thus even though a high training accuracy can be reached, the patterns learned in the end are less useful and less generalizable as they are in L=3.

Overall we conclude that a medium depth network works best when increasing the size of the filters.
"""

part5_q4 = r"""
**Your answer:**

In the first part we clearly see that the best performing model is L=8. Compared to L=8, K=32 from exp1_1 we can see that the ResNet model performs better on the test set while performing worse on the training set. This implies that the skip connections in the ResNet allow the model to learn better even in this depth and thus generalizing better on test data and not overfitting on training data. Furthermore, we can see that the ResNet model is able to train L=16, K=32 while the normal CNN wasn't able to, which also implies that the ResNet helps with the vanishing gradients problem.

Overall compared to exp1_1, we see the ResNet improves on the normal CNN even for deeper layers.

To explain why L=8 performs better than the deeper ResNet models (L=16,32) we consider exp 1_3 where we saw that 3-4 layers are optimal for these models on this dataset. This implies that L=8 and above are simply too deep and would perform worse than L~3-4 which is why out of all the ResNet models, L=8 is the best one. The comparison isn't optimal because here we don't change K but it helps justify the result. 

The same explanation could be used for the second part where we see L=2 perform better with K of length 3, because other models are too deep (>6 layers).

Finally comparing between part 1 and part 2 of exp1_4 we see that even the deeper and more complex ResNet models (L=2, K=64,128,256 is the best one in this part) still underperform compared to the simpler ones (L=8, K=32) which means the skip connections wasn't able to improve enough to justify having deeper and more complex models.

Overall the experiments conclude that for this dataset, the best accuracy is by using ResNet models with medium depth and medium complexity because they provide the best tradeoff between simplicity and generalization (For shallow networks) and deepness and overfitting (or even untrainability when considering vanishing gradients for deep networks).
"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1. The model did not detect the objects in the pictures well. 
 - for 'DolphinsInTheSky' picture, it detected 3 objects' places correctly but failed at classifying them - it classified 2 
of the dolphins as people (one with a semi-high confidence of 0.9 and the other with a low confidence of 0.5) and the other dolphin as 
a surfboard (with a low confidence of 0.37).
- for 'cat-shiba-inu-2' picture, it detected 3 objects places correctly (the dogs) but it classified 2 of the dogs as cats
and one as a dog (with low confidences of 0.66, 0.39, 0.51). The detected shape of the dog that was classfied
correctly was too big. Also, it missed detecting the cat as an object.

2. First, the model failures are reflected by the low confidence scores - the model was uncertain in classifying the objects.
Specific reasons for the model failures can be:
- the model has not trained on enough dolphins/dogs/cats examples.
To resolve this issue, we need to train the model on more examples of dolphins/dogs/cats with 
a good balance between examples of different classes. 
- the objects' features in the picture were not strong. For example, the dolphins were filmed with a backlight and became black.
To resolve this issue, we should apply adjustments to the picture such as color correcting and scaling. 
- the objects in the picture have a similar appearance. For example, shiba inu dogs look similar to cats.
to resolve this issue, we need to train the model on diverse examples of dolphins/dogs/cats - different
appearances, lighting, background etc.

3. To conduct a PGD attack, I would take a sample image and iteratively modify it to maximize the model's prediction error.
Similarly to the attack from tutorial 4, this is done by computing the gradient of a negative loss with respect to the sample
(negating effectively turns the optimization problem into maximizing the loss) 
then updating the image in the direction of the gradient, and then projecting to ensure it remains 
within a specified norm. The iterative updates create an adversarial example that causes the model 
to misclassify objects or miss detections.
"""


part6_q2 = r"""
**Your answer:**


"""


part6_q3 = r"""
**Your answer:**

The model did not detect the objects in the pictures well.

- 'cat_behind_leaves': demonstrates occlusion. The leaves block part of the cat's face, hiding cat features crucial 
for the model to detect a cat. The model did not detect any object in the picture.
- 'dog_at_night': demonstrates illumination conditions. This image is taken at night with low light. The model detected the object's position correctly
but classified the dog as a bird with low confidence (0.27). In the low light only part of the shape of the dog is
visible, making it difficult for the model to detect dog features.
- 'speeding_car': demonstrates blurring. This is a blurred image of a speeding racing car. The model detected 2 objects in the area of the car.
The larger object includes most of the car body, and it was classified as a car with low confidence (0.71). The smaller
object includes the front of the car, and was classified as a car with low confidence (0.31). The double detection
of a car is probably because the blurring deformed the shape and boundries of the car.

"""

part6_bonus = r"""
**Your answer:**

"""