

## MNIST Database

Wikipedia:
>The **MNIST database** (_Modified [National Institute of Standards and Technology](https://en.wikipedia.org/wiki/National_Institute_of_Standards_and_Technology "National Institute of Standards and Technology") database_[[1]](https://en.wikipedia.org/wiki/MNIST_database#cite_note-1)) is a large [database](https://en.wikipedia.org/wiki/Database "Database") of handwritten digits that is commonly used for [training](https://en.wikipedia.org/wiki/Training_set "Training set") various [image processing](https://en.wikipedia.org/wiki/Image_processing "Image processing") systems.[[2]](https://en.wikipedia.org/wiki/MNIST_database#cite_note-2)[[3]](https://en.wikipedia.org/wiki/MNIST_database#cite_note-3) The database is also widely used for training and testing in the field of [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning").[[](https://en.wikipedia.org/wiki/MNIST_database#cite_note-4)

> This set has been studied so much that it is often called the “hello world” of machine learning:
whenever people come up with a new classification algorithm they are curious to see how it will perform on MNIST, and anyone who learns machine learning tackles this dataset sooner or later.


## `OpenML.org`

- OpenML (Open Machine Learning) is an open science / collaborative platform for sharing datasets, machine learning tasks (i.e. experiments setups), algorithms (flows), and experiment runs/results. 
- It’s designed to help reproducibility, benchmarking, comparison of methods, and accelerating ML research by reducing duplicated effort.
- You can think of it as a library + a laboratory: a library of shared ML resources + tools to run experiments, share them, compare, etc. 

---
## The `sklearn.datasets` Package

Scikit-Learn provides the **`sklearn.datasets`** package, which contains functions for accessing and generating datasets. These functions fall into three main categories:
### Functions
#### 1. **`fetch_*` functions**
- Example: `fetch_openml()`
- Used to **download real-world datasets** from external repositories (e.g., [OpenML](https://www.openml.org/)).
- Requires internet connection.

#### 2. **`load_*` functions**
- Example: `load_iris()`, `load_digits()`
- Load **small toy datasets** that are bundled with Scikit-Learn.
- These are available **offline** and don’t require downloading.
#### 3. **`make_*` functions**
- Example: `make_classification()`, `make_regression()`
- Generate **synthetic datasets** for testing or experimenting with algorithms.
- Typically return results in the form of an `(X, y)` tuple:
    - `X`: Input features (NumPy array)
    - `y`: Target labels (NumPy array)

---

### Return Types
Depending on the dataset, the functions return either:
1. **Tuple `(X, y)`**
    - Common for generated (synthetic) datasets.
2. **`sklearn.utils.Bunch` object**
    - A dictionary-like object whose keys can be accessed like attributes.
    - Typically includes the following entries:
        - **`DESCR`** – A description of the dataset.
        - **`data`** – Input data (usually a 2D NumPy array).
        - **`target`** – Labels/targets (usually a 1D NumPy array).

---


## Some Python and Jupyter Introspection Tips 

When working in **Jupyter Notebook**, there are built-in tricks for _exploring objects_:
### Checking Types
```python
type(mnist)         # tells you the type of an object
```
### Listing Attributes / Methods
```python
dir(mnist)          # lists everything you can do with mnist
```
### Getting Documentation
```python
help(mnist)         # detailed docstring if available
```

### In Jupyter specifically:
- `mnist?` → Quick info (docstring).
- `mnist??` → More detailed info (often source code if available).
### Example: methods on a string

```python
s = "hello"
print(dir(s))       # lists all string methods
print(s.upper())    # use one of them
```

### Inspecting Objects Quickly

```python
isinstance(mnist, dict)  # check if it's a dict-like
hasattr(mnist, "data")   # check if it has an attribute
```

---
### Other Useful Tools
- **`vars(obj)`** → Returns `__dict__` of object (attributes).
- **`getattr(obj, "attr_name")`** → Access attribute dynamically.
- **`callable(obj)`** → Check if something can be called like a function.
- **Tab Completion (Jupyter)** → Type `mnist.` then press `<Tab>` to see available methods/attributes.

---

🔑 **Summary for Notes**
- Use `.keys()` and `list()` to inspect datasets like `mnist`.
- In Python, use `type()`, `dir()`, `help()`, `?` / `??` in Jupyter to explore unknown objects.
- Tab completion is your friend for fast discovery.



## About MNIST Train and Test Sets

### 1. General Aspects to Consider When Creating a Test Set
When designing a test set, you want it to **mimic real-world unseen data** while avoiding bias. Some key considerations:
- **Representativeness**
    - The test set should reflect the same distribution as the data you’ll encounter in practice.
    - Example: If you’re building a digit recognizer, your test set must have digits 0–9 in realistic proportions.
- **Independence**
    - Test data should not overlap or be too similar to the training data.
    - Avoid data leakage (e.g., nearly identical images appearing in both sets).
- **Size**
    - Must be large enough to provide statistically reliable evaluation.
    - Too small → high variance in results. Too large → wastes data you could train on.
- **Consistency**
    - If the dataset will be used for benchmarking (like MNIST), the test split should be **fixed and shared** so that results are comparable across studies.
- **Fairness**
    - Should not give the model hints about the evaluation (e.g., same subjects in train and test).
    - Balance across classes (no skew toward one label).
### 2. What the MNIST Creators Considered
For MNIST (LeCun et al.), here’s what they actually did when defining the **60k training / 10k test split**:
- **Balanced classes**
    - Both training and test sets include roughly the same number of each digit (0–9).
- **Independence from training data**
    - The 10,000 test digits come from **different writers** than the training set, so the model must generalize to new handwriting styles.
- **Sufficient test size**
    - 10,000 samples → large enough to give stable error rate estimates (e.g., a 1% error means ~100 misclassifications, which is statistically meaningful).
- **Fixed benchmark**
    - The split is standardized so that every researcher evaluates on the exact same test set → ensures fair comparison.
- **Cleaning & preprocessing**
    - Digits were size-normalized and centered → reducing noise that could otherwise bias the evaluation.

---

## Why Shuffling Matters
- **Shuffling = randomizing the order of your data points.**
- In MNIST, this means that the images of digits are mixed up instead of grouped (e.g., all the “0”s, then all the “1”s, etc.).
### Benefits of Shuffling
1. **Fair cross-validation folds**
    - Cross-validation splits the training set into folds (subsets).
    - If the data wasn’t shuffled, you might end up with some folds missing certain digits.
    - Example: If the first 5,000 rows were all “0”s and “1”s, one fold might contain only those digits → terrible for training and evaluation.
2. **Avoid order sensitivity**
    - Some learning algorithms (e.g., **stochastic gradient descent, online learning**) learn from data **in the order they receive it**.
    - If the first few thousand examples are too similar (e.g., all “7”s), the model may “bias” itself too much toward one class early on.
    - Shuffling ensures that the learning algorithm sees a **diverse mix** of classes throughout training.

✅ In MNIST, since the dataset is already shuffled for you, you don’t need to worry about this.

---
## When Shuffling Is a Bad Idea

Not all data should be shuffled!

- **Time series data** (e.g., stock prices, weather, heart rate monitoring):
    - The order of data points contains essential information (past → future).
    - Shuffling destroys the temporal relationship, which is crucial for prediction.
    - Example: Predicting tomorrow’s stock price requires yesterday’s data, not a random mix of all years’ data.

👉 In such cases, you usually split into **train/validation/test chronologically** (e.g., train on Jan–Jun, validate on Jul, test on Aug–Sep).

---

## 1. Some Notes on Cross-Validation 


### Cross-Validation Works Only on the Training Set
- You start by splitting your full dataset into:
    - **Training set** → used for training & model selection.
    - **Test set** → set aside, untouched, for final evaluation.
- Cross-validation is a **resampling strategy inside the training set**:
    - Split training set into _k_ folds.
    - Train on _k–1_ folds, validate on the remaining fold.
    - Repeat _k_ times so every fold serves as validation once.
    - Average the scores → estimate of how well the model performs.

✅ This helps you:
- Tune hyperparameters (e.g., learning rate, depth of a tree).
- Compare different models fairly.
- Get a less biased estimate of performance than using one fixed validation split.
---
### 2. Why Do We Still Need a Test Set?

Think of the test set as your **final exam**:
- During cross-validation, you’re **still “looking” at the data** in some sense, because you use validation scores to make choices (hyperparameters, feature selection, algorithm).
- This means your model selection process can **overfit the validation folds** (indirectly).
- To get a **truly unbiased estimate**, you need a dataset that was **never touched** during model training _or_ model selection.

That’s the **test set**.
- It answers: _“How will my final chosen model perform on brand-new data?”_

---
### 3. The Workflow in Practice
1. **Train/Validation Phase** (inside training set):
    - Use cross-validation on the training data.
    - Choose the best model and hyperparameters.
2. **Final Evaluation** (on the test set):
    - Train the model again on the _entire training set_ (using the chosen hyperparameters).
    - Evaluate once on the test set.

👉 This final test performance is the number you report.

---
### 4. Simple Analogy
- **Training set** = learning for the exam.
- **Validation folds (cross-validation)** = practice exams (you use them to see how well you’re doing and adjust your study strategy).
- **Test set** = the actual final exam (no peeking beforehand, otherwise it’s cheating).

---

✅ **So the point:**
- Cross-validation = helps choose & tune your model, but can still “leak” bias.
- Test set = unbiased, untouched benchmark of real-world performance.

## Simplifying to a Binary Classifier
Instead of trying to classify all digits (0–9) at once, let’s **narrow the task** to detecting just one digit — for example, the digit **5**.
In this case, the model’s job is simply:
- **Class 1:** the digit is a **5**
- **Class 0:** the digit is **not a 5** (any other digit)

This setup is known as a **binary classification problem**, because the classifier must decide between **two possible outcomes**: _positive_ (5) or _negative_ (not 5).

---
## Creating the Target Vectors

To turn this into a binary problem, we need to **modify the labels**:
```python
y_train_5 = (y_train == '5')   # True for all 5s, False for others
y_test_5  = (y_test == '5')
```
- `y_train_5` and `y_test_5` are now Boolean arrays.
- Each entry is `True` if the original label was “5” and `False` otherwise.
- This way, the model can be trained to distinguish between two categories: **5 vs. not-5**.

---

## Why This Step Helps

- Binary classifiers are often easier to understand and debug than multiclass classifiers.
    
- Many algorithms are first explained in the context of binary classification, since extending to multiple classes is usually done by combining multiple binary classifiers.
    
- This approach also makes it easier to study **evaluation metrics** (like precision, recall, ROC curves) that are naturally defined for binary tasks.

---
## Stochastic Gradient Descent (SGD) — Overview
- **Gradient Descent (GD):**  
    A method for minimizing a loss function by updating the model’s parameters in the direction of the **negative gradient** (steepest descent).
    - Standard GD computes the gradient over the **entire training set** before each update.
- **Stochastic Gradient Descent (SGD):**  
    Instead of using all training examples at once, **SGD updates the parameters using one training instance at a time** (or a small batch).
    - This makes it **much faster for large datasets**, because you don’t have to compute the gradient over millions of examples each step.
    - The updates are noisier than standard GD, but this often helps the model **escape local minima**.

---

## Why SGD is Good for MNIST
1. **Large dataset efficiency**
    - MNIST has **60,000 training images**, so computing the gradient over all examples each iteration would be slow.
    - SGD updates the model **incrementally**, one image at a time → much faster.
2. **Online learning capability**
    - Because SGD processes one instance at a time, you can **train continuously as new data comes in**.
    - Useful in real-world applications where data arrives sequentially (streaming data).
3. **Memory efficiency**
    - Standard GD requires holding all training data in memory.
    - SGD can work with mini-batches or single samples → smaller memory footprint.

---

## In Code (Scikit-Learn Example)

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)  # train on all training images
```
- `SGDClassifier` uses SGD under the hood.
- Once trained, it can classify new images as 5 or not-5.

---
### Key Takeaways
- **SGD** = Gradient Descent “one example at a time.”
- **Good for MNIST** because: large dataset, memory efficiency, online learning potential.
- Even though updates are noisy, the algorithm converges well and is widely used in practice.

---

If you want, I can also **draw a simple diagram showing SGD vs standard GD**, so your notes visually explain why processing one instance at a time is faster and more flexible. It really helps remember the concept.


## How Does SGD Work for This Dataset?
### 1. Context: Our Binary Classifier
We have:
- Input data: `X_train` → each row is a 28×28 image flattened into a vector (784 features).
- Target: `y_train_5` → `True` for digit 5, `False` for others.

We want a **linear model** (like a linear classifier) that predicts:

y^=w⋅X+b\hat{y} = w \cdot X + b
- ww = weights for each feature
- bb = bias / intercept
- Output is interpreted as probability or a score for “is this a 5?”

---
### 2. Gradient Descent for Classification
Unlike linear regression (MSE loss), we typically use a **loss function for classification**, e.g.:
- **Hinge loss** (used in SVMs)
- **Log loss / cross-entropy** (used in logistic regression)

The loss measures **how wrong the model’s predictions are** on a training instance.

### SGD Steps for Each Training Instance

For each training example (xi,yi)(x_i, y_i):

1. Compute **prediction**:
    y^i=w⋅xi+b\hat{y}_i = w \cdot x_i + b
2. Compute **loss**:
    L(y^i,yi)L(\hat{y}_i, y_i)
3. Compute **gradient** of the loss w.r.t. weights ww and bias bb:
    - This tells us how to change ww and bb to reduce the loss for this example.
4. **Update weights and bias**:
    w←w−η⋅∇wLw \gets w - \eta \cdot \nabla_w L b←b−η⋅∇bLb \gets b - \eta \cdot \nabla_b L
- η\eta = learning rate (step size)
- Do this **for each instance individually** → that’s the “stochastic” part.

---
### 3. Why This Works for MNIST
- Each image contributes **one small step** toward improving the model.
- Because the dataset is large (60k images), the **noise in individual updates averages out** over many iterations (epochs).
- The model learns to separate **5s from non-5s** by adjusting weights on all 784 pixels.
    
---
### 4. Training Workflow
1. Shuffle training data → avoid biased sequences.
2. Loop over each training image:
    - Compute prediction
    - Compute gradient of loss
    - Update weights/bias
3. Repeat for multiple epochs → model converges.
    
---
### Analogy:

Think of each training image as a **tiny teacher** giving a hint:
- “Hey, weight for this pixel should be a bit higher”
- “This pixel doesn’t matter, decrease its weight”

After seeing all 60k images multiple times, the model has learned a **rule that separates 5s from non-5s**.

---

### 1. Does the model measure **groups of pixels together** or individually?

- In a **linear classifier** (like `SGDClassifier` with default settings):

$$
y^=w_1x_1+w_2x_2+⋯+w_784x_784+b
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_{784} x_{784} + b
$$

- Each pixel xix_i has **its own weight wiw_i**.
- The prediction is a **weighted sum of all pixels**, plus a bias.

✅ So in effect, it **does consider combinations of pixels indirectly**, because the sum aggregates their contributions.

- But **it does not explicitly model interactions** (like "if both pixel 10 and pixel 50 are dark, then it's 5").
- That’s why linear models sometimes struggle with complex patterns — they only combine pixels linearly.
    

---

### 2. How many parameters do we have?
- For MNIST, each image is 28×28 → **784 features/pixels**.
- So yes, **we have 784 weights w1,w2,…,w784w_1, w_2, …, w_{784}** plus **1 bias term bb**.
- Each time we see a training example, SGD updates all 784 weights **slightly**, depending on how wrong the prediction was.
    

---

### 3. What is the result of the function?
- The raw output of a linear model is:

z=w⋅X+bz = w \cdot X + b

- **Interpretation depends on the loss / classifier:**
    1. **Hinge loss / SVM (default for SGDClassifier)**
        - Output zz can be any real number.
        - Prediction rule:
            y^={1if z≥00if z<0\hat{y} = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}
        - So we just check the **sign** — not necessarily a number between 0 and 1.
            
    2. **Logistic regression** (SGDClassifier can use `"log"` loss)
        - Applies **sigmoid** to raw output:
            
            p^=11+e−z\hat{p} = \frac{1}{1 + e^{-z}}
        - Now p^\hat{p} is between 0 and 1 → probability of being 5.
        - You can threshold at 0.5 → True if p^≥0.5\hat{p} \ge 0.5, False otherwise.

---
### 4. Quick Summary

|Question|Answer|
|---|---|
|Are pixel **groups** considered?|Only indirectly — the weighted sum aggregates all pixels, but no explicit interactions.|
|How many parameters?|784 weights + 1 bias. Each weight is updated for every training example.|
|What is the result?|Raw number zz (can be negative or positive). If using logistic loss, it’s transformed to probability 0..10..1. Then threshold to get True/False.|

---

✅ **Intuition:**
- Each pixel **contributes a bit** to the decision.
- SGD adjusts all pixel weights incrementally so that the **weighted sum crosses the decision boundary** for 5 vs non-5.
- The final classifier is **linear**: it’s basically a “pixel importance map” learned from data.


## Dummy Classifier 
A **Dummy Classifier** is a **simple baseline model** used in machine learning to check how well your real model is doing. It **does not actually “learn” patterns** from the data — instead, it uses simple rules to make predictions.

---

### 1. Purpose

- Establish a **baseline performance**.
- If your real classifier cannot beat a Dummy Classifier, it’s a sign that your model or features are not informative.
- It’s useful to **compare against something trivial** rather than just reporting raw accuracy.

---

### 2. Types of Dummy Strategies (Scikit-Learn)

Scikit-Learn provides `DummyClassifier` with different strategies:

| Strategy          | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `"most_frequent"` | Always predicts the most frequent class in the training set. |
| `"stratified"`    | Predicts randomly, respecting the class distribution.        |
| `"uniform"`       | Predicts randomly with equal probability for all classes.    |
| `"constant"`      | Always predicts a constant value you choose.                 |

Example:

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train_5)
y_pred = dummy_clf.predict(X_test)
```
- If `y_train_5` is mostly False (most digits are not 5), `"most_frequent"` will always predict `False`.
- Accuracy might be high by chance if the data is imbalanced, but it **doesn’t actually detect 5s**.

---

### 3. Why Useful

- Provides a **baseline to beat**.
    
- Helps **debug your evaluation pipeline** — if your sophisticated model performs worse than DummyClassifier, something is wrong.
    
- Helps understand **class imbalance effects** (e.g., always predicting the majority class might look good superficially).
    

---

✅ **Intuition:**  
Think of DummyClassifier as a “random or naive guesser.” Your real model should **always perform better** than this baseline — if not, your features or model are not useful.



## Sample Code For Cross Validation 
```python
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone

  

skfolds = StratifiedKFold(n_splits=3)  # add shuffle=True if the dataset is not

                                       # already shuffled

for train_index, test_index in skfolds.split(X_train, y_train_5):

    clone_clf = clone(sgd_clf)

    X_train_folds = X_train[train_index]

    y_train_folds = y_train_5[train_index]

    X_test_fold = X_train[test_index]

    y_test_fold = y_train_5[test_index]

  

    clone_clf.fit(X_train_folds, y_train_folds)

    y_pred = clone_clf.predict(X_test_fold)

    n_correct = sum(y_pred == y_test_fold)

    print(n_correct / len(y_pred))
```

### Why not just create a new SGDClassifier?

You **could** write:

```python
for fold in folds:
    clf = SGDClassifier(random_state=42)
```

- But `clone(sgd)` is convenient because:
    
    - It **copies all hyperparameters** from the existing `sgd` object (`loss`, `alpha`, `learning_rate`, etc.).
        
    - You don’t have to manually repeat them each time.
        

So `clone(sgd)` is just a **clean, easy way to make a fresh model with the same settings**.

---

###  Does `clone(sgd)` differ from `sgd`?

Yes and no:

|Property|Original `sgd`|`clone(sgd)`|
|---|---|---|
|Hyperparameters|same|same|
|Learned parameters (`coef_`, `intercept_`)|present if already `.fit()`|**reset** — starts fresh|
|Random state|copied|same random state seed (for reproducibility)|

✅ The **clone does not remember anything learned** by `sgd` — it’s a fresh, untrained model.

---

###  What happens when you call `clone_clf.fit()`?

- **All parameters are reset**.
    
- The clone **starts training from scratch** using only that fold’s training data.
    
- Nothing from the original `sgd.fit(X_train, y_train_3)` carries over.
    

---

## Why this is important

- Cross-validation requires training a **new model on each fold**.
    
- If you used the original `sgd` (already trained on the full data), then:
    
    - You would **not be testing the model’s ability to generalize** on unseen data within that fold.
        
    - Cross-validation would be meaningless.
        
- Using `clone()` guarantees:
    
    1. Each fold is independent.
        
    2. Training is only on that fold’s training split.
        
    3. Validation is truly “unseen” data.
        
#### ✅ Key Intuition
- `sgd` = your **base template** (hyperparameters set).
- `clone(sgd)` = **fresh copy**, same hyperparameters, no memory of previous training.
- Each fold → train a **new clone**, test it → measure performance.

---

## `fit` vs `partial_fit`

### 1. Standard `fit()` Behavior

```python
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd.fit(X_train2, y_train2)
```

- By default, **Scikit-Learn’s `SGDClassifier.fit()` resets the model parameters each time**.
    
- So calling `fit()` a second time **does not continue training from the previous weights**.
    
- It’s equivalent to training a fresh model again.
    

---

###  2. How Online / Incremental Training Works

- Scikit-Learn’s `SGDClassifier` supports **online learning** through the **`partial_fit()`** method.
    
- `partial_fit()` **does not reset the parameters** — it continues training from the current model state.
    

Example:

```python
sgd = SGDClassifier()
sgd.partial_fit(X_train_batch1, y_train_batch1, classes=[0,1])
sgd.partial_fit(X_train_batch2, y_train_batch2)
```

- First batch trains from scratch.
    
- Second batch updates the same weights based on new data.
    
- This is **true online learning**, useful for streaming data or very large datasets.
    

---

###  3. Why `fit()` and `partial_fit()` differ

| Method          | Resets weights? | Use case                                              |
| --------------- | --------------- | ----------------------------------------------------- |
| `fit()`         | ✅ Yes           | Standard batch training on a dataset you already have |
| `partial_fit()` | ❌ No            | Incremental/online training on new batches or streams |

---
#### ✅ Key Takeaways
- `sgd.fit()` → trains from scratch, forgets old parameters.
- To train **continuously**, you must use `partial_fit()`.
- This is why **SGD is suitable for online learning**: each new mini-batch can be used to update weights incrementally without retraining from scratch.

---



## Why accuracy can be misleading
We’ll use a **Dummy Classifier** that always predicts the **most frequent class**. In this case, most images are **not 5**, so the dummy model always predicts “non-5”:

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))  # prints False: no 5s detected
```

- Can you guess its accuracy? Let’s check:
    

```python
from sklearn.model_selection import cross_val_score

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# Output: array([0.90965, 0.90965, 0.90965])
```

- That’s right — **over 90% accuracy**, even though this model **never detects a 5**!
    
- Why? Because only about **10% of the images are 5s**, so always guessing “non-5” is correct roughly 90% of the time.
    

✅ This demonstrates an important point: **accuracy can be misleading**, especially on **skewed datasets** where one class dominates.

- A better way to evaluate a classifier is to use a **confusion matrix**, which shows **true positives, false positives, true negatives, and false negatives** — giving a clearer picture of performance on all classes.
    


## About `cross_val_predict`
### 1. What `cross_val_predict` Does

```python
y_train_pred = cross_val_predict(sgd, X_train, y_train_3, cv=3)
```

- `cross_val_predict` **performs cross-validation**, just like `cross_val_score`, but instead of returning **scores**, it returns **predicted labels for each training instance**.
    
- Each prediction is made **only by a model that has never seen that instance during training**.
    

---

### 2. How It Works With 3 Folds

- Suppose `cv=3`. Then `StratifiedKFold` splits the training set into 3 folds. For each fold:
    
    1. Train on 2 folds
        
    2. Predict on the remaining fold (the validation fold)
        
- After looping over all folds, **every training example has a predicted label** from a model that did **not see it during training**.
    

---

### 3. Important Points

- The returned `y_train_pred` is **not an average**.
    
    - It contains **exactly one prediction per instance**, coming from the fold where it was part of the validation set.
        
- This allows you to:
    
    - Evaluate metrics **on the training set** without bias from overfitting.
        
    - Compute confusion matrices, precision, recall, etc., **as if you had a “fresh” model** for each example.
        
- In short:
    
    - Each example is predicted by a model trained **without that example**.
        
    - No averaging is involved; it’s a **direct prediction per instance**.
        

---

### 4. Visualization Example

|Fold|Training Folds|Validation Fold|Predicted Labels Stored For|
|---|---|---|---|
|1|Fold 2 + Fold 3|Fold 1|Predictions for Fold 1|
|2|Fold 1 + Fold 3|Fold 2|Predictions for Fold 2|
|3|Fold 1 + Fold 2|Fold 3|Predictions for Fold 3|

- At the end, `y_train_pred` has predictions for **all instances in X_train**, each from a model that **never saw that instance during training**.
    

---

✅ **Key takeaway:**  
`cross_val_predict` = “give me predictions for each training example, but always from a fold where that example was **excluded from training**.”
- Not averaged
- Not biased by seeing the instance

---
## Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_3, y_train_pred)
cm
```

The **confusion matrix** provides a lot of information about a classifier's performance, 
but sometimes we want a more concise metric. One useful measure is the **accuracy of positive predictions**, 
called the **precision** of the classifier:

$$
\text{precision} = \frac{TP}{TP + FP} \tag{3-1}
$$

- **TP** = number of true positives  
- **FP** = number of false positives  

A trivial way to achieve perfect precision is to build a classifier that always predicts negative, 
except for **one positive prediction** for the instance it is most confident about. 
If that single prediction is correct, precision would be:

$$
\text{precision} = \frac{1}{1} = 100\%
$$

Obviously, such a classifier is **not useful**, since it ignores almost all positive instances.  

---

To address this, precision is usually considered **together with another metric called recall**, 
also known as **sensitivity** or the **true positive rate (TPR)**:

$$
\text{recall} = \frac{TP}{TP + FN} \tag{3-2}
$$

- **FN** = number of false negatives  
- **Recall** tells us how many of the actual positives were correctly detected.  

Together, precision and recall give a more complete picture of classifier performance, especially 
for **imbalanced datasets**.


### Specificity (True Negative Rate)

**Definition:**  
Specificity measures how well a classifier identifies the **negative class** correctly.  
It answers the question:  
➡️ *"Out of all the actual negatives, how many did the model correctly classify as negative?"*

**Formula:**  
$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

- **TN (True Negatives):** negatives correctly predicted as negative.  
- **FP (False Positives):** negatives incorrectly predicted as positive.  

**Interpretation:**  
- High specificity means the model makes **few false positive errors**.  
- It is especially important in cases where **false alarms are costly**.  

**Example Scenarios:**  
- **Medical testing:** If you test for a rare disease, high specificity means healthy people are *not falsely diagnosed*.  
- **Spam filters:** High specificity means important emails are *not wrongly classified as spam*.  

## Harmonic Mean

The **harmonic mean** is a type of average that is especially useful when dealing with **rates, ratios, or percentages**, 
where you want to give more weight to **smaller values**.  

For two numbers \(a\) and \(b\), the harmonic mean is defined as:

$$
H = \frac{2}{\frac{1}{a} + \frac{1}{b}}
$$

- More generally, for \(n\) numbers \($x_1$, $x_2$, $\dots$, $x_n$\):

$$
H = \frac{n}{\frac{1}{x_1} + \frac{1}{x_2} + \dots + \frac{1}{x_n}}
$$

---

### Example in Classification: F1 Score

In machine learning, the **harmonic mean** is used to combine **precision** and **recall** into a single metric, called the **F1 score**:

$$
F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

- Why harmonic mean?  
  - The harmonic mean is **more sensitive to low values**.  
  - If either precision or recall is low, the F1 score will be low.  
  - This prevents a model from looking good if it has very high precision but extremely low recall (or vice versa).

---
### Intuition

- Arithmetic mean = treats all values equally  
- Harmonic mean = **penalizes extreme small values**, emphasizing balance  
- Ideal for **precision & recall**, **speed/efficiency rates**, and other ratios.

# Performance Measures - Complete This
> Evaluating a classifier is often significantly trickier than evaluating a regressor.

##  Measuring Accuracy Using Cross-Validation



## F1 Score
The **F1 score** favors classifiers that balance **precision** and **recall**.  
However, this balance is **not always desirable**, depending on the application.  

- **Precision-focused scenarios:**  
  - Example: detecting videos that are safe for children.  
  - You would prefer **high precision**, even if recall is low.  
  - It is better to reject some good videos than to allow unsafe ones.  
  - A human review pipeline can complement the classifier for borderline cases.  

- **Recall-focused scenarios:**  
  - Example: detecting shoplifters in surveillance footage.  
  - You would prefer **high recall**, even if precision is low.  
  - Security staff can handle some false alerts, but you don’t want to miss actual shoplifters.  

✅ **Key point:** Depending on the context, you may **prioritize precision or recall** over their harmonic mean (F1 score). F1 is most useful when you want a **balanced trade-off** between the two.

## Precision and Recall Tradeoff

**Concept:**  
- Classifiers often output a **score** (e.g., probability or decision function value) for each instance.  
- To turn this into a prediction, you set a **decision threshold**:  
  - Instances above the threshold → classified as **positive**  
  - Instances below the threshold → classified as **negative**

**Effect of threshold choice:**  
- **Higher threshold** → fewer positives predicted  
  - ✅ Precision generally increases (fewer false positives)  
  - ❌ Recall decreases (more false negatives)  
- **Lower threshold** → more positives predicted  
  - ✅ Recall increases (fewer false negatives)  
  - ❌ Precision decreases (more false positives)  

**Key point:** Precision and recall are in tension:  
- Improving one usually hurts the other.  
- This balance is called the **precision–recall trade-off**.

**In Scikit-Learn:**  
- You cannot directly set the decision threshold.  
- But you can access the **raw decision scores** (via `decision_function()` or `predict_proba()`)  
- With these scores, you can **manually adjust the threshold** to find the trade-off that suits your application.





---
## Difference Between `[:-1]` and `[::-1]`
- `[:-1]` → take everything **except the last element**.
    `[1, 2, 3, 4][:-1]  # [1, 2, 3]`
- `[::-1]` → take everything but **reverse the order**.
    `[1, 2, 3, 4][::-1]  # [4, 3, 2, 1]`

They are often used in plotting or preprocessing, but they mean **very different things**:
- `[:-1]` = trim off the last element.
- `[::-1]` = flip the array.

---

## Decision Scores and Thresholds

**How classifiers decide:**  
- Many classifiers (like `SGDClassifier`) don’t just give you a hard prediction (`True/False`).  
- Internally, they compute a **decision score** for each instance:  
  - Positive score → tends to be classified as positive  
  - Negative score → tends to be classified as negative  
- The classifier then compares this score against a **threshold** to make the final prediction.  

By default, `SGDClassifier` uses a **threshold of 0**:  
```python
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)  # e.g., array([2164.22030239])

threshold = 0
y_pred = (y_scores > threshold)  # True → classified as "5"
```
- If the score is **greater than 0**, the model predicts **positive** (`5`).
- If it is **less than 0**, it predicts **negative** (`not 5`).
    

You can change the threshold manually:
`threshold = 3000 y_pred = (y_scores > threshold)  # now returns False`

📌 **Key effect:**
- **Raising the threshold** → classifier becomes more _strict_: higher **precision**, lower **recall**.
- **Lowering the threshold** → classifier becomes more _lenient_: higher **recall**, lower **precision**.

### Choosing the Right Threshold

How do we decide which threshold is best?
1. Use **cross-validation** to get **decision scores** for all training instances:
    ```python
    from sklearn.model_selection import cross_val_predict
    
    y_scores = cross_val_predict(
        sgd_clf, X_train, y_train_5, cv=3,
        method="decision_function"  # important!
    )
    ```
    
    - Here, instead of predictions, we directly collect the **decision scores**.
        
    - This gives us a score for each instance, computed in a _cross-validated way_.
        
2. Use `precision_recall_curve` to compute **precision** and **recall** at all possible thresholds:
    
    ```python
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    ```
    
    - `precisions[i]` and `recalls[i]` correspond to the threshold `thresholds[i]`.
    - An extra point `(precision=0, recall=1)` is added at the end, representing an **infinite threshold**.
        
3. Plot the results to visualize the trade-off:
    
    ```python
    import matplotlib.pyplot as plt
    
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.vlines(3000, 0, 1, "k", "dotted", label="Threshold = 3000")
    
    plt.xlabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()
    ```
#### Why `[:-1]` is Needed

When you call `precision_recall_curve(y_true, y_scores)`, it returns three arrays:  
- `precisions` → precision values for different thresholds  
- `recalls` → recall values for different thresholds  
- `thresholds` → the actual threshold values  

⚠️ Important detail:  
- `precisions` and `recalls` have **one more element** than `thresholds`.  
- The last precision is set to 0, and the last recall is set to 1, corresponding to an **infinite threshold** (no positive predictions at all).  

So, when plotting precision/recall **against thresholds**, you must ignore this last element:  

```python
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
```
Without `[:-1]`, Python would complain about array size mismatch.


---

### Why the Precision Curve is Bumpier than the Recall Curve
- **Recall**:
    - Recall measures how many actual positives were correctly predicted.
    - As you increase the threshold, you make fewer positive predictions → recall can **only go down** (or stay the same).
    - This makes the recall curve smooth and monotonic.
- **Precision**:
    - Precision measures the fraction of predicted positives that are correct.
    - When you increase the threshold:
        - In most cases, precision improves (fewer false positives).
        - But sometimes, precision can **drop**:
            - Example: Suppose you currently predict 5 positives, with 4 correct (precision = 4/5 = 80%).
            - If you increase the threshold slightly, you might exclude one true positive while keeping a false positive.
            - Now you predict 4 positives, with only 3 correct (precision = 3/4 = 75%).
        - Because of these small drops, the precision curve looks “bumpier” than recall.
            

✅ **Key insight:**
- Recall is **monotonic** (always decreases or stays flat with higher threshold).
- Precision is **non-monotonic** (can go up or down depending on which instances get removed).
---

### Summary

- `decision_function()` → gives you **raw decision scores**.
- By setting your own **threshold**, you can control precision vs recall.
- `cross_val_predict(..., method="decision_function")` → lets you get decision scores in a **cross-validated** manner, instead of plain predictions.
- `precision_recall_curve()` → helps you pick the threshold that best matches your application (e.g., prioritize recall for medical screening, or precision for child-safe content filtering).



## Precision–Recall Curve for Trade-off Selection

Another useful way to pick a good balance between **precision** and **recall** is to plot **precision directly against recall** (instead of against the threshold).  

- On this plot, each point corresponds to a threshold.  
- You can visually inspect where **precision starts to drop sharply** as recall increases.  
- For example, in Figure 3-6:  
  - Precision drops quickly once recall passes about **80%**.  
  - A reasonable trade-off might be chosen just before this drop, e.g., around **60% recall**.  

⚠️ The best choice always depends on your project’s priorities (e.g., prioritize recall in medical screening, or precision in spam filtering).  

---

## The `argmax()` Method in NumPy

**Definition:**  
`numpy.argmax(array)` returns the **index of the first occurrence of the maximum value** in the array.  

**Examples:**  
```python
import numpy as np

a = np.array([1, 3, 7, 7, 2])
print(np.argmax(a))   # 2 (the first "7" at index 2)
```
- If the array has multiple maximum values, only the **first index** is returned.

**Use in ML workflows:**
- Suppose you have a boolean mask (array of `True`/`False` values).
- In NumPy, `True` is treated as `1` and `False` as `0`.
- `argmax()` will return the index of the **first True value**.
    
Example:
```python
mask = np.array([False, False, True, True]) 
print(np.argmax(mask))  # 2 → first True
```
So, in the precision–recall context:
- If you want to select the threshold where recall (or precision) first passes a certain condition, you can build a boolean condition (e.g., `recalls >= 0.6`) and call `.argmax()`.
- This gives you the index of the threshold that first satisfies the condition.
---
✅ **Key takeaways:**
- `argmax()` → index of first maximum (or first True in a boolean array).
- Very useful when you need to find the **threshold index** corresponding to a chosen precision/recall criterion.

## ROC
### Watch the following StatQuest videos:
- https://www.youtube.com/watch?v=4jRBRDbJemM

### ROC Curve
The **Receiver Operating Characteristic (ROC) curve** is a common tool for evaluating binary classifiers.  
It looks similar to the **precision/recall curve**, but instead of plotting precision versus recall, it plots:  

- **True Positive Rate (TPR)** = Recall / Sensitivity  
- **False Positive Rate (FPR)** = 1 – Specificity  

The **FPR** (also called *fall-out*) is the proportion of negatives incorrectly classified as positives:  

$$
FPR = \frac{FP}{FP + TN}
$$

Equivalently:  

$$
FPR = 1 - TNR
$$

Here, the **True Negative Rate (TNR)** is the proportion of negatives correctly classified as negative.  
TNR is also called **specificity**.  

So, the ROC curve essentially plots:  

$$
\text{Recall (Sensitivity)} \quad \text{vs.} \quad 1 - \text{Specificity}
$$

---

### Defined Terms (with synonyms, formula, and short description)

- **True Positive Rate (TPR)**  
  - *Other names*: Recall, Sensitivity, Hit Rate  
  - *Formula*:  
    $$
    TPR = \frac{TP}{TP + FN}
    $$  
  - *Short*: Proportion of actual positives correctly identified.  

- **False Positive Rate (FPR)**  
  - *Other names*: Fall-out  
  - *Formula*:  
    $$
    FPR = \frac{FP}{FP + TN}
    $$  
  - *Short*: Proportion of negatives incorrectly classified as positives.  

- **True Negative Rate (TNR)**  
  - *Other names*: Specificity, Selectivity  
  - *Formula*:  
    $$
    TNR = \frac{TN}{TN + FP}
    $$  
  - *Short*: Proportion of negatives correctly identified.  

- **Specificity**  
  - *Other names*: True Negative Rate (TNR)  
  - *Formula*:  
    $$
    Specificity = \frac{TN}{TN + FP}
    $$  
  - *Short*: Ability of a classifier to correctly reject negatives.  

- **Sensitivity**  
  - *Other names*: Recall, True Positive Rate (TPR)  
  - *Formula*:  
    $$
    Sensitivity = \frac{TP}{TP + FN}
    $$  
  - *Short*: Ability of a classifier to correctly identify positives.  

- **ROC Curve**  
  - *Other names*: Receiver Operating Characteristic Curve  
  - *Definition*: Graph plotting sensitivity (recall) against 1 – specificity (FPR) across thresholds.  
  - *Short*: Shows trade-off between true positives and false positives.  


## PR vs ROC

When choosing between the ROC curve and the precision/recall (PR) curve, the decision depends on your data and priorities:
- Use the **PR curve** when the positive class is rare, or when **false positives** are more costly than false negatives.
- Use the **ROC curve** in more balanced cases, or when both types of errors are equally important.

For example, in the earlier ROC curve (and its corresponding ROC AUC score), the classifier seemed to perform very well. However, this was mostly due to the fact that there were far fewer positive examples (5s) than negative ones (non-5s). The ROC curve can sometimes mask this imbalance.

By contrast, the PR curve tells a more honest story: it reveals that the classifier still has plenty of room for improvement, since the curve is not as close to the top-right corner as it could be (see Figure 3-6 again).


## ROC and ROC AUC
### 1. The benefit of a specific **point on the ROC curve**

Each point on the ROC curve corresponds to a particular **decision threshold**. That point tells you the trade-off between:

- **Recall (TPR, sensitivity)** → how many actual positives you are catching.
    
- **FPR (1 – specificity)** → how many negatives you are mistakenly flagging as positive.
    

So, picking a point on the curve is really about choosing the balance between **catching more positives** and **avoiding false alarms**.

- If you pick a point far toward the **top-left**, you’re getting **high recall with low FPR** (ideal).
    
- If you slide toward the diagonal, you’re trading off performance and approaching “random guessing.”
    

👉 **The benefit** of the “best” point is that it optimizes your chosen balance of sensitivity (recall) and specificity (true negatives). It reflects the threshold that works best for your problem.

---

### 2. What ROC AUC tells you

The **AUC (Area Under the Curve)** summarizes the entire ROC curve into a single number:

- **ROC AUC = 1.0** → perfect classifier (every positive ranked higher than every negative).
    
- **ROC AUC = 0.5** → completely random classifier (the diagonal line).
    
- **ROC AUC < 0.5** → worse than random (you could just flip the predictions).
    

In plain words:

- **AUC is the probability that the classifier ranks a randomly chosen positive higher than a randomly chosen negative.**
    
- The higher the AUC, the more separable your two classes are.
    

So while an individual **point** gives you one possible threshold choice, the **AUC** gives you a global view of the classifier’s overall discrimination ability.

---

✅ Example:

- If Classifier A has ROC AUC = 0.95, and Classifier B has ROC AUC = 0.75 → A is generally better, no matter what threshold you choose.
    
- But when deploying, you’ll still need to pick an actual threshold, which corresponds to one point on the ROC curve.
    

---

Would you like me to also show you **how ROC AUC relates back to PR curves**? (since they sometimes give different impressions, especially on imbalanced datasets).


## About `proba` and more
### 1. Why `RandomForestClassifier` has no `decision_function()`
- Some classifiers (like **linear models**: Logistic Regression, SGDClassifier, SVMs) compute a **decision score** — usually the raw distance from the decision boundary (e.g., margin from the hyperplane). That’s what `.decision_function()` gives you.
    
- But **Random Forests** don’t work like that. They don’t have a single linear boundary — they combine votes from many decision trees.
    
- So instead of a raw “score,” what Random Forests can naturally provide is a **class probability** (e.g., 80% chance of being positive, 20% negative).
    
- That’s why they have `.predict_proba()`, not `.decision_function()`.
    

---
### 2. How `predict_proba()` works
When you call `predict_proba(X)` on a Random Forest:
- Each decision tree outputs a predicted class distribution (e.g., one tree might say 7 positives, 3 negatives → probability 0.7 for positive).
- The forest averages across all trees to give the **final probability per class**.

So you get something like:

```python
>>> rf_clf.predict_proba([some_digit])
array([[0.15, 0.85]])   # [P(negative), P(positive)]
```

Here the model thinks there’s an 85% chance the digit is the positive class.

---
### 3. Using probability as a "score"
Since probabilities are continuous numbers between 0 and 1, you can use them as “scores” when plotting **ROC curves** or **Precision/Recall curves**.
- If you set threshold = 0.5 → it’s the normal classifier decision.
- If you set threshold higher (e.g., 0.9) → you only predict positive when the model is very confident, which increases **precision** but reduces **recall**.

So probabilities play the same role as decision scores: they let you slide the threshold and explore the trade-off between precision and recall.

---

### 4. Last paragraph explained
> "Scikit-Learn classifiers always have either a decision_function() method or a predict_proba() method, or sometimes both."

This means:
- **Linear models (e.g., SVMs, logistic regression)** → usually have `.decision_function()` (distance from decision boundary).
- **Probabilistic models (e.g., Random Forest, Naive Bayes)** → usually have `.predict_proba()`.
- **Some models (like logistic regression)** → have both, because you can look at the raw logit score (`decision_function`) _or_ the calibrated probability (`predict_proba`).

👉 **Why this matters**: when you want to draw ROC or PR curves, you don’t want hard predictions (0/1). You want continuous scores (either decision values or probabilities), so you can try different thresholds. That’s why Scikit-Learn ensures at least one of these two methods is available.

## About Calibration
### 1. What the book is saying
When a classifier outputs probabilities with `.predict_proba()`, those numbers are **estimates**, not guaranteed true probabilities.

Example:
- Suppose a model predicts that an image is a "5" with probability **0.55**.
- You might think: “If I collect 100 such images, about 55 should actually be 5s.”
- But in practice, maybe **94 out of 100 are 5s**.  
    → The probability estimates are **biased**: they are too low in this case.

So, the model is **not well-calibrated**.

---
### 2. Calibration issues
Models can be:
- **Under-confident**: predicted probability is lower than reality. (Like the 55% → 94% case.)
- **Over-confident**: predicted probability is higher than reality. (E.g., model says 95%, but only 70% are correct.)

This is common in many classifiers:
- **Naive Bayes** → usually _over-confident_.
- **Random Forests** → can be _under-confident_.
- **Logistic Regression** → often well-calibrated.

---

### 3. Why it matters

If you rely on probabilities for decisions (not just 0/1 predictions), bad calibration can hurt:
- In medicine: a model saying “patient has cancer with 30% probability” must mean _about 30 out of 100 such cases really do_. Otherwise, doctors can’t trust the system.
- In finance: a risk model saying “10% chance of default” should reflect reality, or else your decisions (insurance, loans, etc.) are skewed.

---

### 4. How to fix it — calibration

Scikit-Learn’s [`calibration`](https://scikit-learn.org/stable/modules/calibration.html) package provides methods like:
- **Platt scaling** (fits a logistic regression on top of the raw scores).
- **Isotonic regression** (fits a flexible curve to better map scores to true probabilities).

These methods adjust the raw model outputs so that the predicted probabilities **match the actual frequencies** observed.

---

### 5. Visualization

A common way to check calibration is a **reliability diagram** (calibration curve):
- X-axis: predicted probability bins (e.g., 0.1–0.2, 0.2–0.3, …).
- Y-axis: actual frequency of positives in each bin.
- A perfectly calibrated model lies along the diagonal line `y = x`.
    

## About OvO and OvR
### 1. Recap: The problem

We want to classify digits **0–9** (10 classes).  
But some classifiers (like `SGDClassifier` or `SVC`) can only handle **binary classification** directly.  
So we need a strategy to extend them to multiclass.

---

### 2. Two strategies

#### One-vs-Rest (OvR / OvA)

- Train **1 classifier per class**: e.g., a "0-detector" vs. all other digits.
- For a new image, get the **score** from all 10 classifiers.
- Pick the class with the **highest score**.

Total classifiers: **N** = 10.

---

#### One-vs-One (OvO)
- Train **1 classifier for every pair of classes**.
- Example: one classifier to distinguish 0 vs 1, another for 0 vs 2, another for 1 vs 2, etc.
- For **N classes**, number of classifiers:

$$\frac{N \cdot (N-1)}{2}$$

So for MNIST (N=10):

$$\frac{10 \cdot 9}{2} = 45 \text{ classifiers}$$

---

## 3. How OvO works in practice

Let’s say you want to classify one image:
1. The image is passed through all **45 classifiers**.
2. Each classifier produces a “vote” — e.g., the **0 vs 7 classifier** decides "this looks more like a 7".
3. Count votes for each class.
4. The class with the **most wins** is chosen.

So, OvO = "round-robin tournament": every class duels against every other class, and the winner is the one that wins the most duels.

---

## 4. Why use OvO?

- **Pros**:
    - Each classifier only trains on 2 classes at a time → simpler problem.
    - Especially helpful for algorithms that **don’t scale well** with dataset size (like SVMs). Instead of training on all 60,000 digits, each classifier only sees the subset for its 2 classes (e.g., only 0s and 7s).
- **Cons**:
    - Many classifiers (45 for MNIST, 4,950 if you had 100 classes!).
    - Prediction requires running the image through **all classifiers**, which can be slower.

---

✅ **In short**:

- OvR: one classifier per class, compares score across all.
- OvO: one classifier per pair of classes, run all of them, majority voting decides.
- OvO is often used with SVMs; OvR is usually used with linear models like SGD.


## 🔹 One-vs-One vs One-vs-Rest in Scikit-Learn

When you train an `SVC` on a dataset with more than 2 classes, scikit-learn needs to extend the binary SVM to multiclass. It supports two main strategies:

1. **One-vs-One (OvO)**
    - Trains a separate classifier for **every pair of classes**.
    - With KK classes, this makes $\tfrac{K(K-1)}{2}$ classifiers.
    - Example: MNIST digits (K=10) → 45 classifiers.
    - Each classifier votes for a winner; the class with the most votes is predicted.
    - In case of ties, scikit-learn adds a tiny adjustment (±0.33) using the decision function margin to favor the classifier with higher confidence.
        
2. **One-vs-Rest (OvR)**
    - Trains one classifier **per class vs all others**.
    - With KK classes, this makes KK classifiers.
    - Example: MNIST digits → 10 classifiers.
    - Each classifier outputs a score; the class with the highest score is chosen.
        
👉 You can **force the strategy**:

```python
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

# Force OvO
ovo_clf = OneVsOneClassifier(SVC()).fit(X_train, y_train)

# Force OvR
ovr_clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
```

If you just use `SVC` directly, scikit-learn defaults to **OvR**, but you can change it with `decision_function_shape="ovo"`.

---
## 🔹 Scaling in MNIST (Why it matters for SVMs)

MNIST data consists of **raw grayscale pixel values** between **0 and 255**. If you feed these directly to an SVM:
- Features with **larger numeric ranges dominate** the optimization problem.
- The SVM kernel (RBF, polynomial, etc.) is very sensitive to feature scale because distances in feature space get distorted.
- This leads to poor decision boundaries and lower accuracy.
    

### What scaling does
- Typically, scikit-learn uses `StandardScaler`:
    $x' = \frac{x - \mu}{\sigma}$
    which **centers each feature** (mean = 0) and **scales to unit variance**.
- Alternatively, `MinMaxScaler` rescales features to a fixed range (e.g., [0, 1]).  
    This is often used for pixel data like MNIST.
    

### Effect on MNIST
- Without scaling: ~85.8% accuracy (SVM struggles).
- With scaling (e.g., dividing all pixel values by 255, or using `StandardScaler`): ~89.1% accuracy.
- Scaling makes optimization converge faster and gives the kernel a well-proportioned feature space.

👉 Example:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(float))
X_test_scaled  = scaler.transform(X_test.astype(float))

svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)
```

So the key idea is: **scaling does not change the relative information in MNIST images, it just puts all features on the same numeric footing**, which dramatically helps SVMs.

---

✅ In short:
- **OvO** = many classifiers, each pair of classes, voting + tie-break adjustment.
- **OvR** = one classifier per class vs rest.
- **Scaling** = standardizing pixels so the SVM kernel sees distances fairly, which improves accuracy by several percent.


## 🔹 What a confusion matrix shows
A **confusion matrix** counts how many times the model predicted each class compared to the actual (true) class.
For MNIST digits (0–9):
- Rows = **true labels**
- Columns = **predicted labels**
    
Example (raw counts):

|True \ Pred|0|1|2|3|4|
|---|---|---|---|---|---|
|**0**|950|2|3|0|1|
|**1**|0|1100|5|2|0|
|**2**|4|7|980|15|2|
|**3**|0|3|20|950|5|
👉 This shows counts, but notice: if class **1** has **1100 samples**, while class **0** has only **956 samples**, the matrix is skewed — larger classes look darker in a heatmap, even if accuracy is the same.

---
## 🔹 Why normalize?

Normalization **divides each row by the total number of true samples in that row**:

$$M_{norm}[i, j] = \frac{M[i, j]}{\sum_j M[i, j]}$$

So instead of absolute counts, you get **percentages (fractions) of correct vs incorrect predictions for each class**.

Example (normalized):

| True \ Pred | 0     | 1     | 2     | 3     | 4     |
| ----------- | ----- | ----- | ----- | ----- | ----- |
| **0**       | 0.993 | 0.002 | 0.003 | 0.000 | 0.001 |
| **1**       | 0.000 | 0.993 | 0.004 | 0.002 | 0.000 |
| **2**       | 0.004 | 0.007 | 0.978 | 0.010 | 0.002 |
| **3**       | 0.000 | 0.003 | 0.021 | 0.969 | 0.005 |

Now each row sums to 1.

---

## 🔹 Why is this useful?

- ✅ **Balances class sizes**  
    If one digit appears more often (like "1" in MNIST), its row will naturally have larger counts. Normalization removes this imbalance.
- ✅ **Highlights relative difficulty**  
    You can compare across classes fairly:
    - If digit **5** has only 800 samples but a raw confusion matrix shows 100 misclassifications, that looks small compared to digit **1** with 1500 samples and 150 misclassifications.
    - But normalized: **5 → 100/800 = 12.5% error**, **1 → 150/1500 = 10% error** → actually **5 is harder for the model**.
- ✅ **Makes heatmaps visually comparable**  
    In raw counts, frequent classes dominate the color intensity. With normalized values, the heatmap shading directly reflects **error rates per class**.

---


✅ In short:

- Raw confusion matrix = counts (skewed by class size).
- Normalized confusion matrix = fractions (per-class error rate).
- This helps compare performance across classes fairly, especially if the dataset is imbalanced.

---

![](Pasted%20image%2020250920114922.png)
>But be careful how you interpret the percentages in this diagram: remember that
we’ve excluded the correct predictions. For example, the 36% in row #7, column #9
does not mean that 36% of all images of 7s were misclassified as 9s. It means that 36%
of the errors the model made on images of 7s were misclassifications as 9s. In reality, only 3% of images of 7s were misclassified as 9s, as you can see in the diagram on the
right in Figure 3-9.

>It is also possible to normalize the confusion matrix by column rather than by row:
if you set normalize="pred", you get the diagram on the right in Figure 3-10. For
example, you can see that 56% of misclassified 7s are actually 9s.

>Analyzing the confusion matrix often gives you insights into ways to improve your
classifier. Looking at these plots, it seems that your efforts should be spent on reduc‐
ing the false 8s. For example, you could try to gather more training data for digits that
look like 8s (but are not) so that the classifier can learn to distinguish them from real
8s. Or you could engineer new features that would help the classifier—for example,
writing an algorithm to count the number of closed loops (e.g., 8 has two, 6 has one,
5 has none). Or you could preprocess the images (e.g., using Scikit-Image, Pillow, or
OpenCV) to make some patterns, such as closed loops, stand out more.

![](Pasted%20image%2020250920115628.png)
?


## Data Augmentation
### 🔹 What is Data Augmentation?

**Data augmentation** is the process of **artificially expanding your training dataset by creating modified versions of your existing samples**.  
Instead of just training on the original data, you expose the model to slightly altered variations, so it learns to generalize better.

---

### 🔹 Why is it useful?

Models, especially image classifiers, can be **sensitive to small changes** (like shifts, rotations, scaling, or lighting). If you only train on perfectly centered digits, for example, the model might fail on a digit that’s slightly rotated or off-center.

Data augmentation helps by teaching the model:

- _“A rotated 3 is still a 3.”_
    
- _“A shifted 5 is still a 5.”_
    

This reduces **overfitting** and makes the classifier **more robust to real-world variations**.

---

### 🔹 Example with 3 vs. 5 confusion

As you mentioned:
- The difference between “3” and “5” is **subtle** (the position of the small joining line).
- If a “3” is slightly shifted or rotated, it may look like a “5,” causing misclassification.

Instead of manually fixing rotation/centering for every digit (which is hard), we just **augment** the training data:
- Shift digits left/right/up/down
- Rotate slightly (e.g., ±15°)
- Scale a little bigger or smaller
    
By training on these variations, the model learns to focus on **core shapes** instead of being overly sensitive to exact positioning.

---
### 🔹 Types of Data Augmentation (for images)
- **Geometric transformations:** rotation, translation (shifting), scaling, flipping, cropping.
- **Color/lighting changes:** brightness, contrast, saturation, hue adjustments.
- **Noise injection:** adding Gaussian noise, blurring, sharpening.
- **Random erasing/cutout:** masking parts of the image so the model learns robustness.
- **Synthetic examples:** generating entirely new samples using techniques like GANs.

---
### 🔹 Benefits
- **More data, without collecting more data**
- **Better generalization** (reduces overfitting)
- **Robustness to real-world distortions** (shifts, noise, lighting changes)

---

✅ **In short:**  
Data augmentation is like “teaching by simulation.” By exposing your model to realistic distortions of the data, you make it **less fragile** and **better at recognizing patterns**, even under imperfect conditions.




## What is Multilabel Classification?

In **binary** classification → each instance has **one label** (`cat` or `dog`).  
In **multiclass** classification → each instance has **one label from many** (`0–9 digit`).  
In **multilabel** classification → each instance can have **multiple labels simultaneously**.

👉 Example (digits):
- A digit `9` → labels: `[large=True, odd=True]`
- A digit `8` → labels: `[large=True, odd=False]`
- A digit `4` → labels: `[large=False, odd=False]`

So, multilabel classification = predicting **a vector of labels per instance** instead of just one.

---
### 🔹 Why do labels sometimes depend on each other?

In your MNIST example:

- Label 1: `is_large` (digit ≥ 7)
    
- Label 2: `is_odd` (digit odd?)
    

These two labels are **not independent**:

- If you already know a digit is “large” (7, 8, or 9), then it’s **more likely to also be odd** (since 7 and 9 are odd, only 8 is even).
    
- So `P(is_odd=True | is_large=True)` ≈ 2/3, which is much higher than the global `P(is_odd=True)` (≈ 5/10).
    

This is what the text means when it says:

> "a large digit (7, 8, or 9) is twice more likely to be odd than even."

---

### 🔹 How does KNN handle multilabel here?

The nice thing is:

- **KNN natively supports multilabel classification**.
- It just looks at the nearest neighbors and averages the multiple target labels together.
- Example: If the nearest 5 neighbors are `[1,1,0,1,0]` for the "odd" label, it predicts majority vote = 1 (odd).
- Since it handles all labels **together**, it automatically respects correlations (to some extent).

---

### 🔹 What if your classifier does _not_ support multilabel (like SVC)?

If you use something like `SVC`, it only supports one target at a time.  
Naïve solution: train two separate classifiers:

- One for `is_large`
- One for `is_odd`

⚠️ Problem: These models **don’t talk to each other**.

- The `odd` classifier does not know if the digit was predicted as `large`.
- This loses important label correlations.

---

### 🔹The Chain Method

This is where **classifier chains** come in.

Idea:

- Train models in a **sequence** (a chain).
- Each model predicts one label, but also gets as input:
    - the original features (pixels)
    - plus the predictions of all _previous_ models in the chain.

Example chain:
1. First classifier predicts `is_large` using the pixels.
2. Second classifier predicts `is_odd` using the pixels **and also the output of `is_large`**.

👉 Now the "odd" classifier knows if the model thinks the digit is “large,” and it can adjust its probability accordingly.

So:
- A digit looks like a 7 → first classifier says `large=True` → second classifier updates its odds that it’s `odd=True`.
- Without the chain, the "odd" model would miss this cue.

---

### 🔹Summary

- **Multilabel** = multiple labels per instance (output is a vector).
- **Some classifiers (KNN, RandomForest, LogisticRegression)** can predict multilabels directly.
- **Others (SVC, SGDClassifier)** can’t → need workarounds.
- **Naïve workaround** = train one independent model per label (loses correlations).
- **Chain workaround** = train models sequentially, so later ones use earlier predictions → captures dependencies.

---

✅ **In short:**  
The chain method is needed **only when your base classifier doesn’t support multilabels directly**. It lets each label prediction _benefit from information about the others_.

---
### 🎯 The Core Idea

In practice, the **first few labels act as extra features** for the later classifiers in the chain.  
So, when the model is predicting label #3, it doesn’t just look at the raw input (pixels, embeddings, etc.), it also looks at what it already predicted for labels #1 and #2.

That extra context can shift its probabilities.

---

#### 📌 Example: MNIST with two labels
- **Label 1:** is_large (7, 8, 9 → large, else small)
- **Label 2:** is_odd (1, 3, 5, 7, 9 → odd, else even)

#### Case: The digit is an **8**
1. Classifier #1 (`is_large`) → looks at pixels → predicts `True` (yes, it’s large).
2. Classifier #2 (`is_odd`) → now gets two inputs:
    - Pixels of the digit
    - The prediction `is_large=True`
    
    If it only had the pixels, it might give a 50/50 probability between odd/even (since 8 is even but looks close to 9).  
    But knowing the digit is “large” changes the odds: among large digits, **2/3 are odd** (7, 9 vs 8).
    
    So the model leans more toward predicting `odd=True`.  
    (It might make a mistake here, but that’s how dependencies are captured.)

---

#### 📌 More Labels → Stronger Chains

Suppose you had more labels like:
- `has_closed_loops` (8 has two, 6 has one, 5 has none)
- `top_half_open` (e.g., 5 is open on top, 3 is not)

Now, by the time you predict `is_odd`, the classifier also knows things like:
- “The digit has two closed loops” → likely an 8
- “The digit is large” → must be 7, 8, or 9

All of this **extra structure** makes the prediction more accurate than if each label were predicted blindly.

---

### 🔑 Practical Benefit

In practice, knowing earlier labels:
- **Restricts possibilities** → later classifiers can rule out options.
- **Shifts probabilities** → e.g., if “is_large=True,” then “is_odd” has a higher baseline chance.
- **Adds semantic context** → later predictions don’t start from scratch; they build on earlier ones.

---

## Ways of Evaluating Multi-Label Classifiers
Evaluating multi-label classifiers is trickier than single-label because each sample can have multiple correct outputs. Different metrics capture different aspects of performance, so the “right” choice depends on the application.

---
#### 1. **Per-Label Binary Metrics**
You can treat each label as its own binary classification problem and compute metrics like **precision, recall, and F1** for each label individually.
- Then, average them in different ways:
    - **Macro average** → unweighted mean across labels (treats all labels equally).
    - **Weighted average** → weighted by label frequency (gives more importance to common labels).
    - **Micro average** → aggregates contributions of all labels (good when label imbalance is strong).

Example in Scikit-Learn:

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average="macro")    # treats all labels equally
f1_score(y_true, y_pred, average="weighted") # weights by label frequency
f1_score(y_true, y_pred, average="micro")    # aggregates globally
```

---

#### 2. **Subset Accuracy (Exact Match Ratio)**
Measures the fraction of samples where **all labels are predicted correctly**.
- Very strict metric — a single wrong label makes the entire prediction incorrect.
- Good for tasks where you must get the full label set right (e.g., medical diagnoses).

---

#### 3. **Hamming Loss**
Measures the fraction of **incorrect labels** over all labels.
- Lower is better.
- Less strict than subset accuracy — only penalizes the labels that are wrong.
- Example:
    - True: `[1, 0, 1, 0]`
    - Pred: `[1, 1, 1, 0]` → Hamming loss = 1 wrong label / 4 = 0.25

---

#### 4. **Ranking-Based Metrics**

Useful when classifiers output label scores or probabilities.
- **Coverage Error** → how many top-ranked labels you need to include to cover all true labels.
- **Label Ranking Average Precision (LRAP)** → measures how well the model ranks true labels higher than false ones.
- **One-error** → fraction of samples where the top-ranked label is not actually relevant.

---

#### 5. **Custom Weighting**

In real applications, some labels may be more important than others.

- Example: In face recognition, misclassifying “Alice” might be worse than “Charlie” if you have many more pictures of Alice.
- You can assign custom weights to labels based on domain importance, not just frequency.

---

✅ **Key takeaway:**

- Use **subset accuracy** when exact correctness matters.
- Use **Hamming loss** or **F1 (macro/micro/weighted)** for more flexible evaluation.
- Use **ranking metrics** if you care about the quality of label scores, not just binary predictions.

---


## Micro-Averaging (Precision, Recall, F1)
Micro-averaging works by **flattening all label predictions into one big pool** and then computing the metric globally across all labels.

That means instead of calculating precision/recall/F1 per label and then averaging, it sums up:

- **True Positives (TP)**
- **False Positives (FP)**
- **False Negatives (FN)**

across all labels, and then computes precision/recall/F1 from those totals.

---

#### 🔢 Example

Say you have 3 labels (`A`, `B`, `C`) and 4 samples:

| Sample | True labels | Predicted labels |
| ------ | ----------- | ---------------- |
| 1      | A, B        | A, C             |
| 2      | B           | B                |
| 3      | A, C        | A                |
| 4      | C           | (none)           |

Now compute:

- Across all labels:
    - TP = 3 (`A` in sample 1, `B` in sample 2, `A` in sample 3)
    - FP = 1 (`C` in sample 1)
    - FN = 3 (`B` in sample 1, `C` in sample 3, `C` in sample 4)

Then:

- Precision (micro) = TP / (TP + FP) = 3 / (3+1) = 0.75
- Recall (micro) = TP / (TP + FN) = 3 / (3+3) = 0.5
- F1 (micro) = 2 × (0.75 × 0.5) / (0.75 + 0.5) = 0.6
    
---

#### ⚖️ Why use micro-averaging?

- **Good for imbalanced labels** → common labels contribute more, rare labels less.
- **Focuses on overall correctness** rather than treating each label equally.
- Useful when you care more about **global performance across all labels** than about performance on rare labels.

---

✅ Quick comparison:
- **Macro average**: each label = equal weight (fair to rare labels, but noisy if imbalance).
- **Micro average**: each prediction = equal weight (dominant labels influence more).

---

## Recap of classification types
1. **Binary classification**:  
    One label, two possible classes.  
    Example: “Is this digit a 5?” → Yes/No.
2. **Multiclass classification**:  
    One label, more than two possible classes.  
    Example: “Which digit is this?” → {0,1,…,9}.
3. **Multilabel classification**:  
    Multiple labels, each **binary**.  
    Example: For each digit image, predict:
    - Large (7–9) → Yes/No
    - Odd → Yes/No  
        (Output = `[Large?, Odd?]`)
4. **Multioutput–Multiclass classification** (a generalization):  
    Multiple labels, **each can be multiclass** (not just binary).  
    Example: Image denoising.
    - Input = noisy digit image
    - Output = clean digit image
    - Each pixel = one “label”
    - Each pixel label can take **256 classes** (intensity 0–255)  
        → So the system predicts thousands of labels at once, and each label has multiple possible classes.
        

---

### 🔹 Why is denoising an example?

Think of a **28×28 MNIST image** = 784 pixels.
- In standard classification: we predict **1 digit** (0–9).
- In multioutput–multiclass classification: we predict **784 outputs**, one for each pixel, and each pixel has **256 possible values**.
So instead of one label like “digit=5”, the classifier outputs a **vector of labels**: `[pixel1, pixel2, …, pixel784]`.

---

### 🔹 Key intuition

Multioutput classification is powerful because it handles **structured outputs**:
- Image denoising (predicting a clean image).
- Machine translation (predicting sequences of words).
- Speech-to-text (predicting phonemes or characters).

It’s like saying:

> “Don’t just give me one class per input — give me a whole _grid_ or _set_ of outputs, where each output can itself be multiclass.”

---



## The line between classification and regression
The line between classification and regression is sometimes blurry. For example, predicting pixel intensities could be seen as a regression task rather than classification, since pixel values are continuous (0–255) rather than discrete categories. More generally, multioutput systems are not limited to classification tasks: they can handle any combination of outputs. In practice, this means a single model could simultaneously predict class labels and continuous values for the same input—for instance, an image recognition system that identifies the object in a photo (classification) while also estimating its bounding box coordinates (regression).

---

✅ **Explanation of the last sentence:**

- A **multioutput system** doesn’t need to restrict itself to _only classification_ (e.g., “Is it a cat, dog, or car?”) or _only regression_ (e.g., “Where exactly is the bounding box?”).
    
- It can **mix outputs**:
    - Some outputs are _class labels_ (discrete choices).
    - Some outputs are _value labels_ (continuous predictions).
- Example:
    - Input = a picture of a car.
    - Output 1 = the car’s make/model (classification).
    - Output 2 = the bounding box coordinates (regression).
    - Output 3 = the car’s speed estimate (regression).
    - Output 4 = whether headlights are on/off (classification).
        
So the point is: **multioutput models are flexible — they unify tasks** where you need multiple predictions of different types at once.




# Terms
## Binary Classifier
## DummyClassifer
## Clean Predictions
>
This means that you get a clean prediction for each
instance in the training set (by “clean” I mean “out-of-sample”: the model makes
predictions on data that it never saw during training)

## To address this, precision is usually considered **together with another metric called recall**,  also known as **sensitivity** or the **true positive rate (TPR)**:
## THE SAME FOR PRECISION
## TALK ABOUT NOSTRADAMUS https://en.wikipedia.org/wiki/Nostradamus









