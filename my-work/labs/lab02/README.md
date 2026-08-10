# Lab 02: Train a network by hand

**Module 2: How a neural network learns**

You are going to write a neural network from scratch with NumPy, teach it XOR, and watch the error fall. Nothing here calls a model over the internet, so there is no API key, no account and no bill of any kind. XOR means the answer is 1 when the two inputs disagree and 0 when they agree, and no single straight line can split that pattern, so a network that solves it has genuinely learned a curve. Every model in this course, including the large ones you call over an API later, is built from the same five moves you are about to write yourself: multiply, add, bend, measure the error, nudge.

Before you start, make sure `python llm.py` works. See `setup.html`.

## Steps

### 1. Set up the folder and the two libraries

This lab makes no model calls, so you can ignore the shared llm.py helper for once. Make a folder for the module, then create one empty file called xor_net.py. You will paste each following step onto the end of that same file and re-run the whole file, so by step 8 the file holds the entire network. NumPy is the library that does fast arithmetic on whole tables of numbers at once, and that is what lets a layer of eight neurons be a single line of code instead of a loop. Matplotlib is the library that draws charts, and you need it because most of what you learn in this lab you learn by looking at a picture, not by reading a number. Run the install and expect pip to print either a download line or a message saying the requirement is already satisfied, and both of those are fine.

```python
python -m pip install numpy matplotlib

# then create my-work/labs/lab02/xor_net.py and add the code from step 2 onwards
```

- `python -m pip install numpy matplotlib`: The python -m pip form runs pip using the exact Python you just typed, so the two libraries land in the interpreter you will actually run the script with. Typing plain pip can install into a different Python on the same machine, which is the usual cause of a ModuleNotFoundError two minutes later.
- `# then create my-work/labs/lab02/xor_net.py and add the code from step 2 onwards`: The # makes this a comment, so Python ignores it completely. It is there to tell you the plan: one file, in a folder for this module, grown step by step until it holds the whole network.

> **Watch out:** If import numpy fails right after a successful install, you almost certainly installed into a different Python, so re-run the install with python -m pip from the same terminal you run the script in.

### 2. Make the data, and look at it before you train

XOR means the answer is 1 when the two inputs disagree in sign and 0 when they agree. You build 400 points in four fuzzy clusters, one per corner of a square, and opposite corners share a label. The clusters are fuzzy on purpose, because a network that only ever sees four exact points can memorise those four points, and you want it to learn the shape instead. Always plot your data before training anything, because a picture catches mistakes that a shape check never will. The printed share of 1s should be exactly 0.5, which tells you the two classes are balanced, and that in turn tells you 50% accuracy means the network has learned nothing at all. You should see inputs: (400, 2)  answers: (400, 1)  share of 1s: 0.5, and when you open data.png you will see two colours sitting in diagonally opposite corners.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def make_xor(n=100):
    corners = [(1, 1, 0), (-1, -1, 0), (1, -1, 1), (-1, 1, 1)]
    xs, ys = [], []
    for cx, cy, label in corners:
        xs.append(rng.normal([cx, cy], 0.35, size=(n, 2)))
        ys.append(np.full((n, 1), label, dtype=float))
    return np.vstack(xs), np.vstack(ys)

X, y = make_xor()
print("inputs:", X.shape, " answers:", y.shape, " share of 1s:", y.mean())

plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=12, edgecolor="k", linewidth=0.3)
plt.title("XOR data: opposite corners share a label")
plt.savefig("data.png", dpi=120)
plt.close()
```

- `rng = np.random.default_rng(0)`: This creates a random number generator with a fixed starting point, called a seed. The seed 0 means you get exactly the same random points on every run, which is why the numbers quoted in this lab match the numbers on your screen.
- `corners = [(1, 1, 0), (-1, -1, 0), (1, -1, 1), (-1, 1, 1)]`: Each group of three is a centre x, a centre y, and the label for that corner. Matching signs give label 0 and opposite signs give label 1, which is exactly the XOR rule written as data.
- `rng.normal([cx, cy], 0.35, size=(n, 2))`: This draws 100 points scattered around the corner (cx, cy) with a spread of 0.35 in each direction. Real data is never exact, and that scatter is what forces the network to learn a region of the plane rather than memorise four points.
- `np.full((n, 1), label, dtype=float)`: This makes a column of 100 copies of that corner's label. The shape is (100, 1) and not (100,) on purpose, so it lines up with the network's output column later without NumPy silently reshaping anything.
- `np.vstack(xs), np.vstack(ys)`: vstack stacks the four blocks on top of each other, giving one (400, 2) input table and one (400, 1) answer column. Every later line in this lab assumes exactly those two shapes.
- `c=y.ravel()`: This colours each dot by its label. ravel flattens the (400, 1) column into a flat list of 400 numbers, because scatter wants one colour value per dot and will not accept a column.

**The maths, spelled out**

```
The formula: each point is drawn from a normal distribution, also called a bell curve, written normal(mean, sd).
Symbols: mean is the centre of the bell, here a corner such as (1, 1). sd is the standard deviation, which measures the spread, here 0.35.
Rule of thumb: about 68% of draws land within 1 sd of the mean, 95% within 2 sd, and 99.7% within 3 sd.

Worked example, one corner:
For the corner centred at x = 1 with sd = 0.35, three standard deviations is 3 * 0.35 = 1.05.
So almost every point in that cluster has an x between 1 - 1.05 = -0.05 and 1 + 1.05 = 2.05.
To cross to the wrong side of x = 0 a point must sit (1 - 0) / 0.35 = 2.86 standard deviations below its centre, and the chance of that is about 0.002, which is 2 points in every 1000.

Balance check:
2 of the 4 corners carry label 1, and each corner has 100 points, so the share of 1s is 200 / 400 = 0.5 exactly.

What it means: 0.35 is wide enough that the data looks messy like real data, and narrow enough that the four corners stay clearly separate. If you raised it to 1.0 the clusters would smear into each other and no network could reach 99%.
```

> **Watch out:** The generator rng is shared by later steps, so if you add extra random calls before step 3, or call make_xor twice in one session, your points and every printed number after them will differ from the ones quoted here.

### 3. Write the forward pass

The forward pass is one trip through the network, from the two input numbers out to a single prediction between 0 and 1. W1 is 2 by 8 because you have 2 inputs and 8 hidden neurons, and W2 is 8 by 1 because those 8 neurons feed one output. The starting weights are random on purpose: if you set them all to zero, every neuron in a layer would compute the same thing and receive the same blame forever, so they could never specialise into different jobs. tanh is the bend that stops the two layers collapsing into one, and sigmoid squashes the final number into the range 0 to 1 so you can read it as a probability. The USE_ACTIVATION flag looks pointless now, but step 7 flips it to prove what the bend is worth. Your first printed guesses will be [0.65 0.657 0.666 0.68 0.599] and they mean nothing at all, because the weights are still random, and that is exactly what an untrained network should look like.

```python
USE_ACTIVATION = True

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.tanh(z1) if USE_ACTIVATION else z1
    z2 = a1 @ W2 + b2
    p = sigmoid(z2)
    return a1, p

H = 8
W1 = rng.normal(0, 1, (2, H)) * np.sqrt(1 / 2)
b1 = np.zeros((1, H))
W2 = rng.normal(0, 1, (H, 1)) * np.sqrt(1 / H)
b2 = np.zeros((1, 1))

a1, p = forward(X, W1, b1, W2, b2)
print("first five guesses:", p[:5].ravel().round(3))
```

- `z1 = X @ W1 + b1`: @ is matrix multiply. This one line does all 8 weighted sums for all 400 points at once and hands back a (400, 8) table. b1 has shape (1, 8) and NumPy repeats it down all 400 rows automatically, which is called broadcasting.
- `a1 = np.tanh(z1) if USE_ACTIVATION else z1`: tanh is the bend, and it squashes any number into the range -1 to 1 along a smooth S shape. The else z1 branch passes the number straight through untouched, and that is the switch step 7 uses to remove the bend without editing any other line.
- `p = sigmoid(z2)`: sigmoid squeezes any number into the range 0 to 1, so the output reads as a probability that the answer is 1. Without it the network could output 7.3 or -12, which you cannot compare against a label that is only ever 0 or 1.
- `W1 = rng.normal(0, 1, (2, H)) * np.sqrt(1 / 2)`: This fills W1 with random numbers of average size 1 and then shrinks them by sqrt(1/2), where 2 is the number of inputs feeding each neuron. The shrink stops the weighted sums from starting too large, which would push tanh straight to its flat ends where almost no learning happens.
- `b1 = np.zeros((1, H))`: The biases start at exactly zero. They do not suffer the all-identical problem that zeroed weights do, because the random weights have already made every neuron different from its neighbours.
- `return a1, p`: The function hands back the hidden layer output as well as the prediction. The backward pass in step 5 needs a1 to work out how much each hidden neuron contributed, so throwing it away here would force you to recompute it.

**The maths, spelled out**

```
One neuron: z = x1*w1 + x2*w2 + b
Symbols: x1 and x2 are the two input numbers for one point, w1 and w2 are that neuron's weights, b is its bias.
Worked example with the real first point and the real first hidden neuron from your run:
x = (1.0440, 0.9538) and w = (-0.5205, -0.4991), b = 0.
z = 1.0440 * -0.5205 + 0.9538 * -0.4991 = -0.5434 + -0.4761 = -1.0195
tanh(-1.0195) = -0.7696, and that is neuron 0's output for point 0.

Sigmoid: sigmoid(z) = 1 / (1 + e^-z), where e is about 2.71828.
Worked example: the eight hidden outputs times W2 plus b2 give z2 = 0.617 for that first point.
e^-0.617 = 0.5396, so sigmoid(0.617) = 1 / (1 + 0.5396) = 1 / 1.5396 = 0.6495, which prints as 0.65.
sigmoid(0) = 0.5 exactly, a large positive z goes to 1, a large negative z goes to 0.

Initialisation scale: sd of the weights = sqrt(1 / fan_in)
Symbols: fan_in is how many numbers feed into each neuron of that layer.
Layer 1: fan_in = 2, so sqrt(1/2) = 0.7071.
Layer 2: fan_in = 8, so sqrt(1/8) = 0.3536.
Reason in one line: adding up fan_in independent terms multiplies the spread of the result by fan_in, so dividing the weight variance by fan_in keeps each layer's output roughly the same size as its input.

Parameter count: W1 holds 2 * 8 = 16 numbers, b1 holds 8, W2 holds 8 * 1 = 8, b2 holds 1.
Total = 16 + 8 + 8 + 1 = 33 numbers, and training changes only those 33. Nothing else in the file ever changes.
```

> **Watch out:** USE_ACTIVATION must stay a module-level variable, because forward() reads it fresh on every call, so setting it inside a function without the global keyword silently does nothing at all.

### 4. Score how wrong it is

You cannot improve something you cannot measure, so you crush all 400 mistakes into one single number called the loss. This one is binary cross entropy, which punishes confident wrong answers far harder than unsure ones. When the true answer is 1 the term y * log(p) is all that survives, and it grows painful as p heads towards 0. The np.clip is not decoration: log(0) is negative infinity, and one infinity spreads into every number downstream and turns your whole training run into nan. You should see 0.7429 printed. That is worse than the 0.6931 you would get by answering 0.5 to everything, and that is normal at this point, because the weights are random and random confidence is worse than honest uncertainty.

```python
def loss_fn(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print("loss before any training:", round(loss_fn(p, y), 4))
```

- `p = np.clip(p, 1e-9, 1 - 1e-9)`: This forces every probability into the range 0.000000001 to 0.999999999 before the logarithm ever sees it. sigmoid can return a number that rounds to exactly 0 or 1 in floating point, and log of 0 is negative infinity, which would make the loss nan and destroy every gradient after it.
- `y * np.log(p) + (1 - y) * np.log(1 - p)`: This is an if-statement written as arithmetic. When y is 1 the second term is multiplied by 0 and vanishes, and when y is 0 the first term vanishes, so each point is scored only against its own true answer.
- `-np.mean(...)`: The minus sign flips the score so that lower is better, because the log of any number below 1 is negative. The mean averages over all 400 points, so the loss keeps the same scale even if you later change how many points you have.

**The maths, spelled out**

```
The formula: loss = -(1/N) * sum over all points of [ y*ln(p) + (1-y)*ln(1-p) ]
Symbols: N is the number of points, 400 here. y is the true label, either 0 or 1. p is the network's predicted probability that the answer is 1. ln is the natural logarithm.

Worked example with 4 small points.
True answers y = 1, 1, 0, 0 and predictions p = 0.9, 0.6, 0.2, 0.7.
Point 1: y = 1, score = -ln(0.9) = 0.1054
Point 2: y = 1, score = -ln(0.6) = 0.5108
Point 3: y = 0, score = -ln(1 - 0.2) = -ln(0.8) = 0.2231
Point 4: y = 0, score = -ln(1 - 0.7) = -ln(0.3) = 1.2040
Sum = 2.0433, mean over 4 points = 0.5108
Notice point 4. It was wrong and confident, and on its own it contributed more than the other three added together. That is the whole personality of cross entropy.

The 0.6931 baseline:
If you answer 0.5 for every single point, every score is -ln(0.5) = 0.6931, so the mean is 0.6931.
Treat that as your do-nothing line. Anything above it means your network is doing worse than shrugging.

Your actual first point:
y = 0 and p = 0.6495, so its score is -ln(1 - 0.6495) = -ln(0.3505) = 1.048.
Averaged over all 400 points this comes to 0.7429, which is the number that prints.
```

> **Watch out:** If y ever ends up with shape (400,) instead of (400, 1), NumPy will broadcast it against p into a 400 by 400 table and the loss will still print a believable looking number that is completely wrong.

### 5. Work out the blame and take one step

This is backpropagation, and it is six lines. Backpropagation works out, for each of the 33 numbers in the network, how much the loss would change if you nudged only that number, and it gets all 33 answers by pushing blame backwards from the output rather than by testing each weight one at a time. dz2 = (p - y) / N is the error at the output, divided by the number of points because the loss is a mean. Each line after it carries that blame one stage earlier: through W2 to reach the hidden layer, then through (1 - a1 ** 2), which is the slope of tanh, and finally onto W1. Then you subtract a small fraction of each gradient from its weight, which is one gradient descent step. The loss should tick down from 0.7429 to 0.7219 on that single step, which is a small move and exactly what one nudge should give you.

```python
def backward(X, y, a1, p, W2):
    N = X.shape[0]
    dz2 = (p - y) / N
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * (1 - a1 ** 2) if USE_ACTIVATION else da1
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = backward(X, y, a1, p, W2)
lr = 0.5
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2
_, p = forward(X, W1, b1, W2, b2)
print("loss after ONE step:", round(loss_fn(p, y), 4))
```

- `dz2 = (p - y) / N`: This is the whole error signal at the output, and it is this simple because the sigmoid and the cross entropy cancel each other out when you differentiate them together. Dividing by N is there because the loss is an average and not a sum, so the gradient has to be averaged too.
- `dW2 = a1.T @ dz2`: A hidden neuron that was strongly active gets a bigger share of the blame, so you multiply the error by that neuron's output. The .T transposes a1 from (400, 8) to (8, 400) so the answer comes out shaped (8, 1), which is the same shape as W2 and therefore usable by the update line.
- `db2 = dz2.sum(axis=0, keepdims=True)`: The bias is added to every point equally, so its gradient is just the errors summed down all 400 rows. keepdims=True stops NumPy collapsing the result into a flat array, holding it at shape (1, 1) so it matches b2.
- `da1 = dz2 @ W2.T`: This sends the output error backwards through the second layer of weights. A hidden neuron joined to the output by a large weight receives a large share of the blame, which is exactly the rule you would guess by hand.
- `dz1 = da1 * (1 - a1 ** 2)`: 1 - tanh(z)^2 is the slope of tanh at that point, and blame only passes backwards in proportion to that slope. A neuron sitting near -1 or +1 has an almost flat slope, so it barely learns, and that condition is called saturation.
- `W1 -= lr * dW1`: This is the only line in the step where anything is actually learned. You move each weight a little way in the downhill direction, and lr = 0.5 decides how far a little way is.

**The maths, spelled out**

```
Where dz2 comes from.
For one point the loss is -[ y*ln(p) + (1-y)*ln(1-p) ] and p = sigmoid(z2).
d(loss)/dp = -(y/p) + (1-y)/(1-p)
d(p)/dz2 = p * (1 - p), which is the slope of sigmoid.
Multiply the two and every term cancels down to just (p - y). Then divide by N because the loss averages over N points.
Worked example: the first point has y = 0 and p = 0.6495, with N = 400.
dz2 = (0.6495 - 0) / 400 = 0.001624
It is positive, which says raising z2 would raise the loss, so the update pushes z2 down and p towards 0. That is correct, because the true answer for that point is 0.

The tanh slope.
tanh'(z) = 1 - tanh(z)^2, and since a1 already holds tanh(z1) you can write it as 1 - a1**2 and recompute nothing.
Worked example: neuron 0's output for point 0 is a1 = -0.7696.
slope = 1 - (-0.7696)^2 = 1 - 0.5923 = 0.4077, so about 41% of the blame gets through.
If a1 were 0 the slope would be 1 and all the blame passes.
If a1 were 0.99 the slope would be 1 - 0.9801 = 0.0199, so only 2% passes and that neuron is nearly stuck.

One full chain for one weight, all real numbers from your run.
dz2 = 0.001624, and W2 for hidden neuron 0 is 0.083.
da1 = 0.001624 * 0.083 = 0.0001348
dz1 = 0.0001348 * 0.4077 = 0.0000550
That is one point's contribution. The matrix multiply adds up all 400 of them in a single operation.

The update rule: new_weight = old_weight - lr * gradient
Worked example: a weight of 0.70 with a gradient of 0.02 and lr = 0.5 becomes 0.70 - 0.5 * 0.02 = 0.69.
```

> **Watch out:** This block takes exactly one step every time the file runs, so if you accidentally run it twice the printed loss after ONE step will not be 0.7219 and every number after it drifts.

### 6. Loop it, and plot the loss falling

An epoch is one pass over all your data, and this loop does 3000 of them, using all 400 points every single time. Training is now just five things on repeat: forward, record the loss, backward, subtract, go again. Note that train() builds its own fresh weights from a seed, so it does not continue from the weights you made in steps 3 and 5, it starts over from scratch. You should see the loss start at 0.6994 and finish at 0.0136, with the printed checkpoints falling to 0.0316 by epoch 500 and then crawling from there. Open loss.png and you will see a curve that drops like a cliff and then flattens into a long tail. That flattening is normal and not a bug, because once the easy points are right the remaining gradient is small, so each step moves the weights less.

```python
def train(X, y, H=8, lr=0.5, epochs=3000, seed=0, quiet=False):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 1, (X.shape[1], H)) * np.sqrt(1 / X.shape[1])
    b1 = np.zeros((1, H))
    W2 = r.normal(0, 1, (H, 1)) * np.sqrt(1 / H)
    b2 = np.zeros((1, 1))
    history = []
    for epoch in range(epochs):
        a1, p = forward(X, W1, b1, W2, b2)
        history.append(loss_fn(p, y))
        dW1, db1, dW2, db2 = backward(X, y, a1, p, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        if not quiet and epoch % 500 == 0:
            print(f"epoch {epoch:5d}  loss {history[-1]:.4f}")
    return (W1, b1, W2, b2), history

params, history = train(X, y)
print("final loss:", round(history[-1], 4))

plt.plot(history)
plt.xlabel("epoch (one pass over all the data)")
plt.ylabel("loss (lower is better)")
plt.title("The loss falling")
plt.savefig("loss.png", dpi=120)
plt.close()
```

- `r = np.random.default_rng(seed)`: The function makes its own generator with its own seed instead of borrowing the global rng. That is what makes every call to train() start from identical weights, so when you change the learning rate in step 8 the learning rate really is the only difference.
- `W1 = r.normal(0, 1, (X.shape[1], H)) * np.sqrt(1 / X.shape[1])`: X.shape[1] is the number of input columns, read from your data rather than hard-coded as 2. That is why the same train() will work unchanged on the mini-project data at the end without you touching a line.
- `history.append(loss_fn(p, y))`: The loss is recorded before the update, so history[0] is the loss of the completely untrained network. Keeping the whole list is what lets you draw the curve instead of only ever seeing the final number.
- `the four -= lines`: These are one gradient descent step applied to all 33 numbers in the network. This is the only place in the entire lab where anything is learned, and everything else exists to feed these four lines.
- `if not quiet and epoch % 500 == 0`: % is the remainder after division, so this is true on epoch 0, 500, 1000 and so on. Printing all 3000 lines would bury the useful output, and the quiet flag lets steps 7 and 8 call train() with no printing at all.

**The maths, spelled out**

```
What one epoch costs here.
This is full-batch gradient descent, meaning every update uses all 400 points, so 3000 epochs is exactly 3000 weight updates.
Total point evaluations = 3000 epochs * 400 points = 1,200,000 forward passes through a 33 number network, which takes a second or two on a laptop CPU.

Why the curve flattens.
The size of each step is lr * gradient, and the gradient at the output is (p - y) / N.
Worked example: a point with y = 1 that the network currently calls p = 0.3 contributes (0.3 - 1) / 400 = -0.00175.
The same point once p has climbed to 0.98 contributes (0.98 - 1) / 400 = -0.00005, which is 35 times smaller.
So the network slows itself down as it gets things right, and no extra code makes that happen.

Reading your own numbers.
The loss falls from 0.6994 to 0.0316 across the first 500 epochs, then only from 0.0316 to 0.0136 across the remaining 2500.
Fraction of the total improvement done in the first 500 epochs = (0.6994 - 0.0316) / (0.6994 - 0.0136) = 0.6678 / 0.6858 = 0.97.
About 97% of the learning happened in the first sixth of the run.
```

> **Watch out:** train() creates brand new weights inside itself, so if you expected it to carry on from step 5 and are puzzled that the loss restarts near 0.70 instead of 0.72, that is the reason.

### 7. Check accuracy, then break it on purpose

Loss is the thing you optimise, but accuracy is the thing you understand, so measure both. Accuracy turns each probability into a yes or a no by asking whether it is above 0.5, then counts how often that matches the truth, and you should get 99.5%. Now set USE_ACTIVATION = False and train the identical network again, with the same starting weights, the same loss and the same learning rate. It parks at 0.6931 and about 50.2% accuracy, which is the coin-flip baseline from step 4. Nothing about the network's size changed, it still has all 33 numbers and all 8 hidden neurons, so the only thing you took away was the bend. That single experiment is the whole argument for activation functions, and it is worth running twice before you believe it.

```python
def accuracy(params, X, y):
    _, p = forward(X, *params)
    return float(((p > 0.5) == (y > 0.5)).mean()) * 100

print("accuracy with tanh:", round(accuracy(params, X, y), 1), "%")

USE_ACTIVATION = False
flat_params, flat_history = train(X, y, quiet=True)
print("no activation -> loss", round(flat_history[-1], 4),
      " accuracy", round(accuracy(flat_params, X, y), 1), "%")
USE_ACTIVATION = True
```

- `_, p = forward(X, *params)`: The star unpacks the tuple (W1, b1, W2, b2) into four separate arguments for forward. The underscore is the usual Python way of saying you are deliberately throwing away the first returned value, which here is the hidden layer you do not need.
- `(p > 0.5) == (y > 0.5)`: Both sides turn into tables of True and False, and == compares them point by point. Writing y > 0.5 instead of y == 1 sidesteps the floating point trap of testing a decimal number for exact equality with a whole number.
- `.mean()) * 100`: In NumPy True counts as 1 and False counts as 0, so the mean of a True/False table is simply the fraction that were correct. Multiplying by 100 turns that fraction into a percentage.
- `USE_ACTIVATION = False ... USE_ACTIVATION = True`: This flips the global flag, retrains the same network with no bend, then flips it back. Putting it back is not optional, because step 8 calls train() again and would otherwise be quietly measuring four straight-line models.

**The maths, spelled out**

```
Why removing the bend costs everything.
With no activation the hidden layer is just a1 = X*W1 + b1, so the output becomes
z2 = (X*W1 + b1)*W2 + b2 = X*(W1*W2) + (b1*W2 + b2)
W1 is 2 by 8 and W2 is 8 by 1, so W1*W2 is a single 2 by 1 matrix. Eight neurons collapse into one straight line, and adding eight hundred more would still collapse into one straight line.

What your run actually learned:
W1*W2 came out as (0.0167, -0.0041) with a combined bias of 0.0004.
So the prediction is sigmoid(0.0167*x1 - 0.0041*x2 + 0.0004).
For a typical point with x1 = 1 and x2 = 1 that is sigmoid(0.0130) = 0.5032, which is 0.5 for all practical purposes.
The loss of answering 0.5 everywhere is -ln(0.5) = 0.6931, exactly the number that printed.

Accuracy arithmetic:
400 points, 201 of them happened to land on the correct side of a boundary that is essentially noise.
201 / 400 = 0.5025, which rounds to the 50.2% you see.

Being honest about one detail: a straight line placed by hand could reach 75% on XOR by slicing off one corner. Gradient descent never finds it, because this data is balanced in every direction, so for any line the mirror-image line scores exactly the same, and the only stable point the optimiser can settle on is the do-nothing solution in the middle.
```

> **Watch out:** Forgetting the final USE_ACTIVATION = True is the classic mistake here, and the symptom is that all four curves in step 8 come out flat and equally useless.

### 8. Change only the learning rate

Same data, same network, same 3000 epochs, one number different each time. The learning rate is the fraction of the gradient you actually subtract from each weight, and it is the setting people get wrong more often than any other. You should see 0.001 finish at 0.6860, which means it never really started, 0.05 at 0.0477, 0.5 at 0.0136, and 500 at 0.8073, which is worse than an untrained network. The log scale on the y axis is there so the good curve and the terrible curve both fit on one chart, because 0.0136 and 0.8 are too far apart to read on an ordinary scale. The huge rate may also print a RuntimeWarning about overflow in exp, which is a symptom of numbers blowing up rather than a crash, and it is another sign your step size is wrong. No formula hands you the right value, so you try several, plot them together, and keep the one whose curve falls fastest and stays down.

```python
for lr_try in [0.001, 0.05, 0.5, 500.0]:
    _, h = train(X, y, lr=lr_try, quiet=True)
    plt.plot(h, label=f"lr={lr_try}")
    print(f"lr={lr_try:<8} final loss {h[-1]:.4f}")
plt.legend()
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Same network, four learning rates")
plt.savefig("learning_rates.png", dpi=120)
plt.close()
```

- `for lr_try in [0.001, 0.05, 0.5, 500.0]`: These four values span about six orders of magnitude, because learning rates are searched by multiplying, not by adding. Trying 0.001, 0.002 and 0.003 would tell you almost nothing, since they all behave the same way.
- `_, h = train(X, y, lr=lr_try, quiet=True)`: Only lr changes between runs, and the fixed seed inside train() guarantees identical starting weights every time. quiet=True suppresses the per-epoch printing so you get four clean summary lines instead of twenty-four.
- `plt.plot(h, label=f"lr={lr_try}")`: Each curve is drawn onto the same figure, and the label is the text that plt.legend() picks up afterwards. There is no plt.figure() call between the plots, which is exactly why all four land on one chart.
- `plt.yscale("log")`: A log y axis puts the same screen distance between 0.01 and 0.1 as between 0.1 and 1. Without it the curve that drops to 0.0136 would be a flat line squashed against the bottom edge and you could not tell the good runs apart.
- `f"lr={lr_try:<8} final loss {h[-1]:.4f}"`: The :&lt;8 pads the rate out to eight characters on the left so the four printed lines line up in the terminal. The :.4f prints exactly four decimal places, which is enough to compare 0.0136 against 0.0477.

**The maths, spelled out**

```
A one-weight model of what is going on.
Imagine the loss is a simple bowl: loss = a * w^2.
Symbols: w is a single weight, a says how steep the bowl is, lr is the learning rate.
The gradient is 2*a*w, so the update is w_new = w - lr*2*a*w = w * (1 - 2*a*lr).
All the behaviour lives in that one multiplier, (1 - 2*a*lr).

Worked example with a = 1 and a starting w = 1.
lr = 0.1 gives multiplier 1 - 0.2 = 0.8, so w goes 1, 0.8, 0.64, 0.512 and settles at 0. Healthy.
lr = 1.0 gives multiplier 1 - 2 = -1, so w goes 1, -1, 1, -1 forever and never improves.
lr = 1.5 gives multiplier 1 - 3 = -2, so w goes 1, -2, 4, -8, 16 and explodes. This is your lr = 500 curve finishing at 0.8073.

The rule: you converge when the multiplier sits between -1 and 1, which means lr must be below 1 / a. You never know a, so you test several values.

Why 0.001 stalls: in this toy bowl 3000 tiny steps do eventually arrive. On the real XOR loss surface the gradients are far smaller than 2*w, so after the same 3000 steps the loss has only crept from 0.6994 to 0.6860, a total move of 0.013.

Why the overflow warning appears: sigmoid computes e^-z, and a float64 number cannot hold anything bigger than about 1.8e308. ln(1.8e308) = 709.78, so the moment -z climbs past roughly 710 the exponential overflows to infinity and NumPy warns you. That only happens once the weights have already blown up.
```

> **Watch out:** If you skipped the plt.close() at the end of step 6, all four learning rate curves get drawn on top of the old loss curve and the chart becomes unreadable.

## You are done when

Your module folder holds three PNG files: data.png, loss.png and learning_rates.png. Your terminal shows a final loss of 0.0136 and the line 'accuracy with tanh: 99.5 %'. The no-activation run prints a loss of 0.6931 and about 50.2%, which is the coin flip. loss.png shows one curve dropping from about 0.70 to under 0.02, and learning_rates.png shows four curves in which lr=0.001 is a nearly flat line across the top and lr=500.0 bounces around above all the rest.

---

## Mini-project: Teach it a new shape

Train the lab's network on two concentric rings, then save the trained weights and both data splits so a program can verify the result instead of you eyeballing it. Your script writes rings_result.json and rings_boundary.png, and check.py reads both.

- Write make_rings(n=500) in a new file rings.py, next to a copy of the lab's forward, loss_fn, backward, train and accuracy functions. Label 0 is a blob with radius drawn between 0.0 and 1.0, label 1 is a ring with radius drawn between 2.0 and 2.5, both at a random angle. Return X shaped (N, 2) and y shaped (N, 1) so train() works with no changes.
- Shuffle with a seeded generator, then split 80/20 into Xtr, ytr and Xte, yte. The training loop must never see the test points, and the checker looks for leaks.
- Train with H=8 and the lab's defaults, then measure accuracy on both splits with the lab's accuracy(). Then train a second network on the same split with H=2 and keep only its test accuracy.
- Draw the boundary: np.meshgrid over roughly -3 to 3 in both directions, run forward() on every grid point, colour the plane with plt.contourf, scatter your held-out points on top. Save it as rings_boundary.png in the same folder.
- Write rings_result.json in that folder with exactly these keys: hidden (8), train_X, train_y, test_X, test_y (nested lists, use .tolist()), train_accuracy and test_accuracy (percentages from 0 to 100), weights (an object with W1, b1, W2, b2, each .tolist(), so b1 is [[...]] with 8 numbers and b2 is [[x]]), and small_net ({"hidden": 2, "test_accuracy": ...}).
- Save check.py in the same folder and run python check.py.

### Check it

`check.py` is in this folder. Run it:

```bash
python check.py, run from the folder holding rings_result.json
```


**You are done when** check.py prints eleven PASS lines and ends with ALL CHECKS PASSED, and exits with code 0. It rebuilds the forward pass from your saved weights and scores your test points itself, so a hand-typed accuracy fails the match check, and a test point that also sits in the training set fails the leak check. It also prints one line saying the picture is not checked automatically: open rings_boundary.png and confirm the coloured region closes into a ring around the inner blob.

**If you want more:** Delete one whole corner of the XOR data, train on the remaining three corners, then test only on the corner you removed. Expect close to 0% correct on it. Work out why before you look anything up, and note what it says about trusting a model on inputs unlike its training data.
