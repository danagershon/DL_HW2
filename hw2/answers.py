r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
        0.01,  # vanilla SGD  # TODO LEFT: tweak values to get better test accuracy than 35.1
        0.01,  # SGD with momentum  # TODO LEFT: tweak values to get far better test accuracy than vanilla
        0.0001,  # RMSProp  # TODO LEFT: tweak more, 0.0001 is better than 0.001, against 0.0005 not sure
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

    # TODO LEFT: find best wstd and lr
    # bad combinations: 0.1,0.01  0.2,0.005  0.05,0.01  0.1,0.006  0.11,0.006
    # good combinations: 0.1,0.005  0.1,0.004  0.09,0.005  0.11,0.005  0.09,0.004 (best)
    
    # to try: 
    combinations = [
        (0.1, 0.005),
        (0.1, 0.004),
        (0.1, 0.006),
        (0.09, 0.005),
        (0.11, 0.005),
        (0.09, 0.004),
        (0.11, 0.006)
    ] 

    # wstd, lr = combinations[6]  # change index to match the combination to test

    wstd, lr = (
        0.09,  # Weight standard deviation
        0.004,  # Learning rate
    )
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    loss_fn = torch.nn.CrossEntropyLoss() #just picked one that looks ok
    lr, weight_decay, momentum = 0.06, 0.04, 0.05  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
- for 'cat-shiba-inu-2' picture, it detected 3 objects places correctly (the dogs) but it classified the dogs as cats
(with low confidences of 0.66, 0.39, 0.51). It missed detecting the cat as an object.

2. First, the model failures are reflected by the low confidence scores - the model was uncertain in classifying the objects.
Specific reasons for the model failures can be:
- the model has not trained on enough dolphins/dogs/cats examples.
to resolve this issue, we need to train the model on more examples of dolphins/dogs/cats with 
a good balance between examples of different classes. 
- the objects' features were not strong. For example, the dolphins were filmed with a backlight and became black.
to resolve this issue, we should apply adjustments to the picture such as color correcting and scaling. 
- the objects in the picture have a similar appearance. For example, shiba inu dogs look similar to cats.
to resolve this issue, we need to train the model on diverse examples of dolphins/dogs/cats - different
appearances, lighting, background etc.

3. To conduct a PGD attack, I would take a sample image and iteratively modify it to maximize the model's prediction error.
Similarly to the attack from tutorial 4, this is done by computing the gradient of a negative loss with respect to the sample,
(negating effectively turns the optimization problem into maximizing the loss) 
then updating the image in the direction of the gradient, and then projecting to ensure the remains 
within a specified norm. The iterative updates create an adversarial example that causes the model 
to misclassify objects or miss detections.
"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

The model did not detect the objects in the pictures well.

- 'cat_behind_leaves': demonstrates occlusion. The leaves block part of the cat's face. The model
did not detect any object in the picture.
- 'dog_at_night': demonstrates illumination conditions. This image is taken at night with low light. The model detected the object's position correctly
but classified the dog as a bird with low confidence (0.27).
- 'speeding_car': demonstrates blurring. This is a blurred image of a speeding racing car. The model detected 2 objects in the area of the car.
The larger object includes most of the car body, and it was classified as a car with low confidence (0.7). The smaller
object included the front of the car, and was classified as a car with low confidence (0.31).

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""