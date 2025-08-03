
# End-To-End
## Pipelines

A **pipeline** is a **sequence of data processing steps** that are chained together. In machine learning, pipelines are crucial because:
- **Data needs multiple transformations** before it becomes usable (e.g., missing value imputation, scaling, feature engineering).
- By chaining transformations and model training into a pipeline, you **ensure consistency** (the same steps are applied to training and future data).
- Pipelines help **reduce code duplication**, **improve readability**, and **simplify cross-validation**.

### Key Concepts:
- **Asynchronous components**: Each component in the pipeline processes data independently and at different times.
- **Data stores as interfaces**: Components communicate by reading/writing to/from data stores. This separation enhances modularity.
- **Resilience**: If a component fails, others can still work using cached or last-available data.
- **Monitoring is essential**: Without it, failures may silently degrade performance as stale or incorrect data propagates.

## Framing the Problem
Before you start training a model, you need to **define the problem clearly**. This is **step zero** in any machine learning project. Misframing the problem leads to building the wrong solution.
### ❓ Ask These Questions:
1. **What type of learning problem is it?**
    - **Supervised Learning**  
        → You have **labeled data** (input-output pairs).  
        Examples:
        - Predicting house prices (regression)
        - Email spam detection (classification)
    - **Unsupervised Learning**  
        → You have **no labels**, only inputs.  
        Examples:
        - Customer segmentation (clustering)
        - Dimensionality reduction for visualization
    - **Reinforcement Learning**  
        → An agent learns by interacting with an environment and receiving **rewards**.  
        Examples:
        - Game playing (e.g., AlphaGo)
        - Robotics control
2. **What type of supervised task is it?**
    - **Classification**  
        Predicting discrete categories (e.g., spam or not spam)
    - **Regression**  
        Predicting continuous values (e.g., predicting temperature)
    - **Multilabel or Multiclass**
        - Multiclass → One label from many (e.g., digit recognition: 0–9)
        - Multilabel → Multiple labels can be true at once (e.g., a movie can be both “comedy” and “drama”)

3. **What kind of learning setup fits best?**
    
    - **Batch Learning (Offline Learning)**
        - Train the model on the **full dataset at once**
        - Model is static after training unless you retrain it from scratch
        - Suitable for large, stable datasets
    - **Online Learning**
        - Model is updated **incrementally** as new data comes in
        - Ideal for streaming data or situations where data evolves over time
        - Useful when computational resources are limited or retraining from scratch is costly

By answering these questions, you define the **type of solution you need**, which in turn influences:

- Your **data collection strategy**
- The **algorithms** you’ll consider
- How you’ll **evaluate performance**
- The **infrastructure** for deployment and retraining

---

## 📘 Notation

This section defines several common machine learning notations used throughout the book.

---

- Let $m$ be the number of instances in the dataset.

    - Example: If the validation set has 2,000 districts, then $m = 2000$.

- Let $\mathbf{x}^{(i)}$ be the **feature vector** of the $i^{th}$ instance (excluding the label).

- Let $y^{(i)}$ be the **label** (target value) of the $i^{th}$ instance.

    - Example:
      $$
      \mathbf{x}^{(1)} =
      \begin{bmatrix}
      -118.29 \\
      33.91 \\
      1416 \\
      38372
      \end{bmatrix}, \quad
      y^{(1)} = 156400
      $$

- Let $\mathbf{X}$ be the **feature matrix**, containing all feature vectors in the dataset:

    - Each row is an instance:
      $$
      \mathbf{X} =
      \begin{bmatrix}
      (\mathbf{x}^{(1)})^\top \\
      (\mathbf{x}^{(2)})^\top \\
      \vdots \\
      (\mathbf{x}^{(m)})^\top
      \end{bmatrix}
      $$

- Let $h$ be the **prediction function** (also called a **hypothesis**).

    - Given an instance's feature vector $\mathbf{x}^{(i)}$, the system predicts:
      $$
      \hat{y}^{(i)} = h(\mathbf{x}^{(i)})
      $$

    - Example:
      If $h(\mathbf{x}^{(1)}) = 158400$, then the prediction error is:
      $$
      \hat{y}^{(1)} - y^{(1)} = 158400 - 156400 = 2000
      $$

- Let $\text{RMSE}(\mathbf{X}, h)$ be the **Root Mean Squared Error** cost function evaluated on dataset $\mathbf{X}$ using hypothesis $h$.

---

### ✅ Typography Conventions

- **Scalars** → lowercase italic: $m$, $y^{(i)}$, $h$
- **Vectors** → lowercase bold: $\mathbf{x}^{(i)}$
- **Matrices** → uppercase bold: $\mathbf{X}$


---


## 📏 Select a Performance Measure: RMSE vs MAE

To evaluate how well a machine learning model performs, we need a way to **measure the distance** between its **predictions** and the actual **target values**. Two common metrics for regression tasks are:

---

### ✅ RMSE (Root Mean Squared Error)

- Measures the **Euclidean distance** between predicted and true values.
- It uses the **$\ell_2$ norm**:
  $$
  \text{RMSE}(\mathbf{X}, h) = \sqrt{\frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)^2}
  $$
- **Sensitive to outliers**: squaring the errors makes large errors count more.
- Preferred when **outliers are rare** and follow a **normal distribution**.

---

### ✅ MAE (Mean Absolute Error)

- Measures the **Manhattan distance** between predictions and targets.
- It uses the **$\ell_1$ norm**:
  $$
  \text{MAE}(\mathbf{X}, h) = \frac{1}{m} \sum_{i=1}^m \left| \hat{y}^{(i)} - y^{(i)} \right|
  $$
- **More robust to outliers** than RMSE. (It is **less affected** by extreme values or **rare, unusually large/small** errors)
- Gives equal weight to all errors, regardless of magnitude.

---

## 🔢 General Form: $\ell_k$ Norm

- For a vector $\mathbf{v}$ with $n$ elements:
  $$
  \| \mathbf{v} \|_k = \left( \sum_{i=1}^n |v_i|^k \right)^{1/k}
  $$
- Special cases:
  - $\ell_0$: number of nonzero elements
  - $\ell_1$: Manhattan norm (MAE)
  - $\ell_2$: Euclidean norm (RMSE)
  - $\ell_\infty$: max absolute value in the vector

---

### 🧠 Insight

- **Higher $k$ values** focus more on **large errors** and ignore smaller ones.
- Use **RMSE** when large errors should be penalized more heavily.
- Use **MAE** when all errors should be treated **equally**.

---

📌 **Tip:** Always plot your error distribution. If you have many outliers or a skewed distribution, **MAE** might give you a more stable picture.


## Check the Assumptions




## 💡 What Does "More Robust to Outliers" Mean?

When we say a metric is **more robust to outliers**, we mean:

- It is **less affected** by extreme values or **rare, unusually large/small** errors.
- These outliers don’t overly influence the metric’s overall value.

For example:
- RMSE **squares the errors**, so **large errors have a disproportionately big impact**.
- MAE **treats all errors equally**, so it’s **more stable** when there are extreme predictions.

**Use MAE** if:
- Your dataset has noise or **unpredictable spikes**.
- You want to **avoid letting a few big errors dominate the evaluation**.

---

## 📒 Notebooks and Google Colab

### What Is a Notebook?

A **Jupyter Notebook** (used in Google Colab) is an interactive coding environment that combines:

-  **Code**
-  **Explanatory text (Markdown)**
-  **Outputs like graphs, tables, etc.**

You can **run cells one at a time**, view results immediately, and mix code with documentation, making it great for **data exploration, ML experiments, and teaching**.

---

### 🔗 Google Colab Shortcuts

- Colab is a free, cloud-based Jupyter notebook environment provided by Google.
- It lets you run Python code without any setup on your local machine.
- You can access many useful keyboard shortcuts.

Here’s a full list of **Colab/Jupyter shortcuts**:  
👉 [Jupyter Notebook Shortcuts – Towards Data Science](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330/)

To view shortcuts inside Colab:
1. Click on the **"Tools"** menu
2. Select **"Keyboard shortcuts"**  
Or press: `Cmd/Ctrl + M + H`

---

### 💾 Accessing Files in Google Colab

When you run a Colab notebook:

- Your Google Drive is mounted under:
  ```
  /content/drive/MyDrive
  ```
- To **save a model or data file** to your Drive:
  ```python
  !cp /content/my_model_file /content/drive/MyDrive
  ```
- The `!` at the start tells Colab to run a **Linux shell command**, not Python code.
- `cp` is the command for **copying files** in Linux.

🛠️ Colab notebooks run on **Linux (Ubuntu)** virtual machines, so basic Linux commands work!

---

### 📦 About Imports

In regular Python scripts:
- It's recommended by **PEP 8 (Python's style guide)** to put all `import` statements **at the top** of the file.

But in notebooks:
- It's common to **import packages where they're needed**, inside individual cells.
- This makes your code more **modular and easier to follow**, especially for tutorials or demos.

Example:
```python
import pandas as pd  # common import for data manipulation
```

This flexibility is part of why notebooks are so popular for **experimentation** and **teaching**.

---

## 📊 Understanding Percentiles

**Percentiles** help describe the **distribution** of a dataset by indicating **how data is spread** across values.

### 🔢 Definitions

- A **percentile** is the value **below which a given percentage** of data points fall.

  For example:
  - The 25th percentile = 25% of the values are **less than or equal to** this number.
  - The 50th percentile = **median** (middle value)
  - The 75th percentile = 75% of values fall **below** this value.
### 📘 Terminology

| Term              | Meaning                              |
|-------------------|---------------------------------------|
| 25th percentile   | First quartile (Q1)                   |
| 50th percentile   | Median (Second quartile, Q2)          |
| 75th percentile   | Third quartile (Q3)                   |





## Why the data has been **transformed**:
- The dataset has **been preprocessed**:
  - The `median_income` values are **scaled** to range between **0.5 and 15**, but not exactly:
    - **Minimum is 0.4999**
    - **Maximum is 15.0001**
- These are **floating-point approximations** and represent:
- `$5,000` → stored as `0.5`  
- `$30,000` → stored as `3.0`  
- `$150,000` → capped and stored as `15.0`  

| Transformation       | Explanation                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Scaled**           | All income values are divided by **10,000** — so income is now in **tens of thousands**                                            |
| **Capped** (clipped) | Extremely low incomes were set to no lower than **~$5,000** (0.5 × 10k), and high incomes to no more than **~$150,000** (15 × 10k) |
### 📌 What Is **Capping**?

**Capping** means **limiting values** in a dataset so they do not go beyond a certain range.
- If a value is **too small**, it's set to a **minimum threshold**.
- If a value is **too large**, it's set to a **maximum threshold**.
This is done to reduce the effect of **outliers** or **extreme values** that can distort analysis or model training.
---
### ✅ Why Use Capping?

| Reason                       | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| **Reduce outlier effect**    | Models like linear regression can be skewed by extreme values. |
| **Avoid division errors**    | In log transforms, very small values can break calculations.   |
| **Keep data in known range** | Some ML algorithms assume features fall in a specific range.   |

### ⚠️ A Tradeoff

While capping improves **stability**, you also **lose information** about how extreme the original values were. That’s okay **if those extremes are noise or rare** — but bad if they’re meaningful.

📌 **Tip:** It’s common to work with **engineered or normalized features** in ML. Just make sure to understand them before jumping into model training.



##  **Why is skewed data harder for ML algorithms?**

### a. Many ML models assume or work better with **linear** or **symmetric** data.
Let’s take **linear regression** as an example:
- It assumes a roughly **linear relationship** between features and target.
- If a feature is **right-skewed**, that relationship may be **nonlinear**, distorted, or dominated by outliers.
- Result: The model might overfit the tail or underperform on the dense (important) part of the distribution.

It's harder to find a clear pattern unless you “pull in” those extreme values.

### b. Optimization algorithms struggle

Models like logistic regression or SVM rely on **gradient descent**, which assumes:
- Features are on **similar scales**,
- Gradients behave **predictably**.

Right-skewed features have:
- **Long tails** with rare, extreme values → they contribute **too much** to the gradients.
- That throws off optimization, making training slower or unstable.

### c. Distance-based models (like k-NN or clustering) become misleading

- Skewed features mean **Euclidean distance** (how "far" two points are) becomes unreliable.
- A single large feature can dominate the entire distance calculation.

---

## What does **"transforming"** the data mean?

> “You can’t just change the distribution, right?”

In ML, **transforming** means applying a **mathematical function** to **reshape the values**, while preserving their **relative order or meaning**.

---

### ✅ Examples of Transformations

#### Log transform
If `x = [1, 10, 100, 1000]`

```python
np.log(x)
→ [0, 2.3, 4.6, 6.9]
```

- Still increasing, still ordered.
- But now: much less skewed, easier to work with.

#### Square root
```python
np.sqrt([1, 4, 16, 100])
→ [1, 2, 4, 10]
```

Again: relative relationships preserved, but large numbers pulled inward.

---



## Test Set

It may feel **premature** to set aside part of your data early on, especially before exploring the dataset in detail. However, it’s a **critical step** for ensuring a reliable machine learning workflow.
### Why Set Aside a Test Set Early?
- Your brain is a **powerful pattern detector**, but that also makes it **prone to overfitting**.
- If you explore the full dataset (including the test set), you might unconsciously:
  - Notice patterns **specific to the test set**
  - Choose or tune models **based on those patterns**

This can lead to:
- **Over-optimistic performance estimates**
- A model that **looks good in testing**, but performs **poorly in the real world**

This problem is known as **data snooping** (or **data leakage**).

---
##  Data Snooping (Bias)

**Data snooping** happens when information from the test set **leaks** into the model selection or training process. This contaminates your evaluation.

### Example:

You might:
- Look at the test set’s distribution
- Notice a correlation
- Decide to use a specific model architecture because of it

Now, when you measure performance on the test set:
- It’s no longer an **unseen**, **independent** measure
- Your result is **biased** — and **over-optimistic**

---

## ✅ Solution: Keep the Test Set Sacred

- Set aside the **test set from the very beginning**
- Don’t look at it until the **very end**, after:
  - Data exploration
  - Feature engineering
  - Model selection
  - Cross-validation on the training set

Use it only once:
- To **evaluate the final model’s generalization performance**

---

📌 **Tip:** You can also create a **validation set** (or use **cross-validation**) during model development, and keep the **test set completely untouched** until the final evaluation.



## `.iloc` vs `.loc` in Pandas

In pandas, both `.iloc` and `.loc` are used to **access rows and columns** in a DataFrame — but they work differently:

---
### 🔢 `.iloc[]` → **Integer Location**

- Access **by position (index number)**
- Purely **integer-based** indexing

#### Example:
```python
df.iloc[0]         # First row
df.iloc[0:3]       # First 3 rows
df.iloc[0, 1]      # Row 0, column 1
```

---

### 🏷️ `.loc[]` → **Label-based Location**

- Access **by label (row/column names)**
- Can use slices, lists, or boolean masks

#### Example:
```python
df.loc[0]              # Row with label/index 0
df.loc[0:3]            # Rows from label 0 to 3 (inclusive!)
df.loc[0, 'name']      # Value at row 0, column 'name'
df.loc[df['age'] > 30] # All rows where age > 30
```

---

### Summary Table

| Feature | `.iloc[]`             | `.loc[]`             |
|---------|------------------------|-----------------------|
| Access by | **Index position**      | **Index/column label** |
| Returns   | Rows/columns by number | Rows/columns by name  |
| Slice     | Excludes end           | **Includes** end      |

---

📌 **Tip:** Use `.iloc` when you're working with **position**, and `.loc` when you're working with **labels**.




## 🌌 The Answer to Life, the Universe, and Everything

In *The Hitchhiker’s Guide to the Galaxy*, a group of hyper-intelligent, pan-dimensional beings builds a **supercomputer** named **Deep Thought** to calculate:

> **“The Answer to the Ultimate Question of Life, the Universe, and Everything.”**

After **seven and a half million years** of computation, Deep Thought finally responds:

> **"42."**

---

### 😐 Wait, what?

The answer — **"42"** — is intentionally **absurd and meaningless without the right question**.

Deep Thought explains:
> “I think the problem, to be quite honest with you, is that you've never actually known what the question is.”

So they build an even greater computer (called **Earth**) to figure out **the actual question**.

---

## What does it mean?

Douglas Adams has said he picked **42** arbitrarily — it’s just a **joke**. It satirizes:

- Our obsession with finding deep, cosmic meaning
- The idea that a single number could explain everything

It’s become a **cultural meme** used to:
- Humorously answer unanswerable questions
- Signal geeky, sci-fi humor

---

📘 **Fun Fact**: In ASCII, `*` is the 42nd character — which some interpret as a nod to "*everything*."

---

> "The answer to life, the universe, and everything is 42."  
> — Deep Thought (and every Hitchhiker’s fan)







## **Does the ML algorithm have memory or state across runs?**

No — **by default, it does not**.

When you **rerun your ML training script**, the algorithm starts from scratch. So technically, **your model doesn't remember anything from previous runs** — you're right about that.

### ✅ So what's the concern, then?

The concern is **you**, the **developer or analyst**.

### 🎯 The Real Problem: **You (and your code) leak test data over time**

Even if the ML model doesn't remember anything across runs:

- **You** might see parts of the test set on one run, make decisions, change preprocessing, tweak model hyperparameters, rerun — and now your model is indirectly influenced by **test set knowledge**.
    
- Over time, you’re **accidentally training your model on patterns that leak from the test set** — defeating its purpose.
    

---

### 📌 Think of it this way:

You want your **test set to simulate real-world, unseen data**.  
But if it keeps changing randomly every time you rerun your code, you and your model will **eventually see the whole dataset**.

So the key principle here is:

> **The test set must remain fixed and untouched across multiple experiments.**

This is why we care about **repeatable splits**.

---

## Why isn't `np.random.seed(42)` enough?

Setting a random seed **is enough for reproducibility** — but **only as long as your dataset doesn't change**.

Here’s what that means:

### ✅ What setting the seed does:

```python
np.random.seed(42)
shuffled_indices = np.random.permutation(len(data))
```

This guarantees the same **shuffled order of indices** every time **for the same dataset**.

So:

- Your train/test split is the same across runs.
    
- You’re reproducible.
    

### ❌ But the problem is:

If you get **new data** (say, a newer version of your dataset with more rows), `np.random.permutation()` will shuffle all data again — and:

- **Previously "test" instances may now end up in training**, and vice versa.
    
- The seed doesn’t protect against **dataset changes**.
    

This defeats the main goal: to **permanently isolate test data** even when new data arrives.

---

## ✅ So what is the better solution?

### Use a **deterministic rule** based on **something that won’t change**, like:

- A unique, stable **ID** (e.g., customer ID, row hash, image filename).
    
- Apply a function like `hash(id)` and assign it to test if it's in the bottom X% of hash values.
    

This way:

- The **same data points** always end up in the test set.
- **New data** gets tested only if it's new.
- You don't leak info from train to test **even when your dataset updates**.

---

## Sampling Bias from Coarse Location Info

### 🔍 What It Means:

The dataset includes **location information** (like latitude and longitude), but this info is **coarse** — meaning it's **not very precise**.

- Multiple **districts share the exact same location values**.


## A possible implementation for using hash
```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
	return crc32(np.int64(identifier)) < test_ratio * 2**32
	
def split_data_with_id_hash(data, test_ratio, id_column):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
	return data.loc[~in_test_set], data.loc[in_test_set]
```


## Creating Unique IDs for Dataset Splitting
- The housing dataset **lacks a built-in unique identifier** column.
- A simple solution is to use the **row index** as an ID by resetting the index:
  ```python
  housing_with_id = housing.reset_index()  # adds an `index` column
  train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
  ```
- If using the row index as ID:
  - Ensure **new data is only appended** (no deletion or reordering).
- If this isn’t feasible, create an ID from **stable features**, e.g., combine latitude and longitude:
  ```python
  housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
  train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
  ```
- Using stable features helps maintain consistent IDs over time.



```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

## Stratified Sampling and Income Categories

- In surveys, maintaining important population ratios (e.g., **48.9% males**) in samples is crucial to avoid bias.
- **Stratified sampling** divides the population into **homogeneous subgroups (strata)** and samples the right number from each, ensuring the test set represents the overall population.
- Purely random sampling can lead to **skewed samples**; for example, a ~10.7% chance of female representation falling outside 48.5–53.5%, biasing results.

---

### 🏦 Handling Continuous Variables: Income Categories

- Median income is a key predictor for housing prices, but since it's continuous, it needs to be **binned into categories** for stratified sampling.
- Most median incomes cluster between **$15,000 and $60,000** (scaled values 1.5 to 6), but some go higher.
- To avoid bias, strata should be:
  - **Few enough** to have sufficient data in each
  - **Large enough** for meaningful representation
- Example code uses `pd.cut()` to create 5 income categories:
  - Category 1: 0 to 1.5 (less than $15,000)
  - Category 2: 1.5 to 3, etc.

This allows creating a stratified test set that accurately reflects income distribution in the population.

## ❓ Should We Stratify by Feature?

**Stratifying** means splitting your dataset so that your **train/test (or validation) sets reflect the distribution of important features** in the overall data.

### When to Stratify:
- **If a feature strongly influences the target variable**, stratifying on it helps ensure:
  - The model **sees all important subgroups** during training.
  - The evaluation on the test set is **more reliable and representative**.
- Common features to stratify on include:
  - Categorical features (e.g., gender, region, class)
  - Discretized continuous features (e.g., income categories, age bins)

> **Discretization** means **converting continuous values into distinct categories or bins**.

### When Not to Stratify:

- If the feature is **not relevant or weakly correlated** with the target, stratifying may add unnecessary complexity.
- If stratification leads to **very small strata**, it might cause unstable splits.

---



## What Is **Stratified Sampling**?

> **Stratified sampling** means splitting your dataset in such a way that the **distribution of a specific variable (usually categorical)** is preserved across the train and test sets.

In other words:
- You group data into **strata** based on some **important feature**, like income category or class label.
- Then you **sample from each stratum proportionally**.
    
---
### 📊 Why do this?

Because **random sampling can break the natural distribution** of important features — especially in **small datasets**.

### With **stratified sampling**, your test set would keep the same proportions:

```text
80% train / 20% test:
Train: 48 low, 24 medium, 8 high  
Test: 12 low, 6 medium, 2 high
```

---

## 🔧 When to Use Stratified Sampling?

✅ Use it when:
- You have **imbalanced classes** (e.g., fraud vs. non-fraud, disease vs. healthy).
- You’re splitting based on **categorical features** (e.g., education level, income group).
- The target variable's distribution matters.

🚫 Not useful when:
- The feature has no meaningful groupings,
- Or is continuous and evenly distributed.
---

##  Stratified in `scikit-learn`

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
```
Here:
- `income_cat` is a categorical version of income (e.g., bucketed into 5 groups).
- The split ensures the same proportions in train/test.

---


```python
from sklearn.model_selection import StratifiedShuffleSplit

  

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
```


```python
strat_train_set, strat_test_set = train_test_split(

    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    
```


---
## ⚠️ Neglected but Critical: Test Set Generation

- Generating a **proper test set** is a **crucial step** in any machine learning project.
- It is often **neglected**, but getting this right helps avoid biases and ensures reliable evaluation.
- The concepts learned here will also be important later for **cross-validation**.
- After preparing the test set correctly, the next stage is **exploring the data**.

---


## Matplotlib plot options



| Argument                        | Meaning                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `kind="scatter"`                | Tells pandas to make a **scatter plot**, not a line/bar plot.                              |
| `x="longitude"`                 | Horizontal axis shows **longitude** (i.e., east-west location).                            |
| `y="latitude"`                  | Vertical axis shows **latitude** (i.e., north-south location).                             |
| `grid=True`                     | Turns on the grid behind the plot (for better readability).                                |
| `alpha=0.2`                     | Transparency (0 = invisible, 1 = opaque). Makes overlapping points visible.                |
| `s=housing["population"] / 100` | Sets **bubble size**: more population = larger circle. Divided by 100 to scale it down.    |
| `label="population"`            | Label for the plot (shows in legend — not always effective with scatter).                  |
| `c="median_house_value"`        | Sets **color** of each point based on house value.                                         |
| `cmap="jet"`                    | The **colormap**: "jet" means low values are blue, high values are red.                    |
| `colorbar=True`                 | Adds a side bar to explain the color scale (house prices).                                 |
| `legend=True`                   | Tries to show a legend (usually not very useful here unless you have a categorical label). |
| `sharex=False`                  | Don’t force axis sharing — only matters in subplot grids.                                  |
| `figsize=(10, 7)`               | Width × height of the figure, in inches.                                                   |

## numeric only
```python
corr_matrix = housing.corr(numeric_only=True)
```


## 📈 Insights from Data Visualization and Correlation

### 1. Observing Scatter Plots and Data Patterns

- The scatter plot shows a **strong positive correlation**: as one variable increases, so does the other, with points fairly close to the trend line.
- A **price cap at $500,000** is clearly visible as a horizontal line, reflecting a limit in the data.
	- Several other **horizontal lines appear at price points** like $450,000, $350,000, and $280,000, indicating **data quirks or artificial limits**.
- To prevent the model from simply learning these quirks, consider **removing or treating these districts separately**.

---

### 2. Understanding the Correlation Coefficient

- The **correlation coefficient** measures **linear relationships** — how much one variable increases or decreases as the other does.
- It **cannot detect nonlinear relationships**, which might be important but invisible to this metric.
- Examples show datasets with **zero correlation coefficient but clear nonlinear patterns**.
- A correlation of **±1 means a perfect linear relationship**, but it says nothing about the slope or units.
  - Example: Height in inches vs. height in feet has a correlation coefficient of 1.

---

### 3. Exploring Attribute Combinations

- Before feeding data into machine learning algorithms, try **combining different attributes** to discover useful features.
- Feature engineering can improve model performance by capturing more complex patterns.

---

📌 **Tip:** Always visualize your data and understand the limitations of simple statistics like correlation before modeling.


## ❓ Why Are There Data Quirks (Horizontal Lines) in the Plot?

- The horizontal lines at certain price values (like $500,000, $450,000, etc.) often happen because of **data capping or rounding**:
  - The dataset might have **upper limits set** on housing prices (e.g., any house over $500,000 is recorded as $500,000).
  - Some prices might be **rounded or grouped** into buckets for privacy or simplification.
- These quirks are **artifacts of data collection or preprocessing**, not natural variations.
- The problem: models may learn these artificial patterns instead of true underlying relationships, hurting generalization.
