## **Definitions**

>Machine Learning is the science (and art) of programming computers so they can
learn from data

>[Machine Learning is the] field of study that gives computers the ability to learn
without being explicitly programmed.
—Arthur Samuel, 1959

>A computer program is said to learn from experience E with respect to some task T
and some performance measure P, if its performance on T, as measured by P,
improves with experience E.
—Tom Mitchell, 1997

## **They have already been here** 
**Machine Learning History (Brief):**
- **1950s:** Birth of ML (Samuel’s checkers program, Rosenblatt’s perceptron).
- **1980s:** Backpropagation & early neural networks revived ML.
- **1990s–2000s:** Statistical models (SVM, Random Forest) became practical with more data.
- **2010s:** Deep learning boom (AlexNet, ImageNet).
- **2020s:** Generative AI & large-scale ML models (Transformers, LLMs).
Although the term _machine learning_ was coined in **1959** by **Arthur Samuel**, its roots go back to early AI research in the 1950s.

### Spam detectors
#### **1. The Earliest Ideas (1996–1998)**
- **Who:**
    - **Markus Sahami** (Stanford University, later at Google) is often credited as the pioneer.
    - His 1998 paper _"A Bayesian Approach to Filtering Junk E-Mail"_ laid the foundation.
- **When:**
    - Research started around **1996–1997**, and his **1998 paper** became the first influential ML-based spam filtering method.
- **How:**
    - Used **Naive Bayes classifiers**, a simple but powerful probabilistic model.
    - The system learned from a dataset of emails labeled as _spam_ or _ham (not spam)_.
    - **Words in the email body** were treated as features. The probability that an email was spam was computed using Bayes’ theorem.
    - Formula (simplified):
        $$
P(\text{Spam} \mid W_1, W_2, \dots, W_n) = 
\frac{P(\text{Spam}) \prod_{i=1}^{n} P(W_i \mid \text{Spam})}
{P(\text{Spam}) \prod_{i=1}^{n} P(W_i \mid \text{Spam}) + P(\text{Ham}) \prod_{i=1}^{n} P(W_i \mid \text{Ham})}
$$


---


## **Where Machine Learning Shines**

Machine Learning excels in problems that are either too complex for traditional approaches or have no known algorithm.

ML can also help **humans learn** by revealing what the model has discovered. For example, after training a spam filter on enough emails, it can be inspected to show which words or word combinations are the strongest predictors of spam (though interpreting some algorithms can be tricky).

Additionally, applying ML techniques to large datasets can uncover patterns that are not immediately obvious—a process known as **data mining**.


## **Typical Use Cases**
- **Problems requiring extensive fine-tuning or long rule lists:**  
    A single ML algorithm can often simplify code and outperform traditional methods.
- **Complex problems with no good traditional solution:**  
    Advanced ML techniques may find workable solutions.
- **Fluctuating environments:**  
    ML systems can continuously adapt to new data.
- **Gaining insights from complex or massive datasets:**  
    ML helps extract meaningful patterns and relationships.
## **Machine Learning Examples**
- **Classifying products on a production line**  
    **Task:** Image classification (e.g., detecting defective items using CNNs).
    
- **Detecting brain tumors in scans**  
    **Task:** Semantic segmentation (pixel-level tumor detection using CNNs).
    
- **Classifying news articles**  
    **Task:** Text classification (e.g., politics vs. sports, using RNNs or Transformers).
    
- **Flagging offensive comments**  
    **Task:** Text classification (e.g., detecting hate speech with Transformers).
    
- **Summarizing long documents**  
    **Task:** Text summarization (e.g., summarizing research papers with BERT or T5).
    
- **Creating a chatbot or personal assistant**  
    **Task:** NLP (NLU + question answering, e.g., Siri, ChatGPT).
    
- **Forecasting company revenue**  
    **Task:** Regression (e.g., predicting sales using Random Forests or RNNs for time series).
    
- **Reacting to voice commands**  
    **Task:** Speech recognition (e.g., voice-controlled smart speakers using RNNs or Transformers).
    
- **Detecting credit card fraud**  
    **Task:** Anomaly detection (e.g., Isolation Forests, Autoencoders).
    
- **Segmenting clients for marketing**  
    **Task:** Clustering (e.g., grouping customers by purchase behavior using K-Means).
    
- **Visualizing high-dimensional datasets**  
    **Task:** Dimensionality reduction (e.g., PCA or t-SNE to visualize gene expression data).
    
- **Recommending products**  
    **Task:** Recommender systems (e.g., Amazon product suggestions using neural networks).
    
- **Building an intelligent game bot**  
    **Task:** Reinforcement Learning (e.g., AlphaGo learning to play Go).


## **Types of Machine Learning Systems**
### Overview
1. **By Supervision**
    - **Supervised Learning:** Trained with labeled data.
        - **Tasks:**
            - **Classification:** e.g., spam filter (spam/ham).
            - **Regression:** e.g., predicting car prices.
        - **Common Algorithms:** k-Nearest Neighbors, Linear/Logistic Regression, SVMs, Decision Trees, Random Forests, Neural Networks.
    - **Unsupervised Learning:** Trained on unlabeled data to find patterns (e.g., clustering, dimensionality reduction).
        
    - **Semi-supervised Learning:** Mix of labeled and unlabeled data.
    - **Reinforcement Learning:** Agents learn through rewards and penalties.
2. **By Learning Mode**
    - **Online Learning:** Learns incrementally as new data arrives.
    - **Batch Learning:** Trained on all data at once, then deployed.
3. **By Approach**
    - **Instance-based Learning:** Compares new data to known examples (e.g., k-NN).
    - **Model-based Learning:** Detects patterns and builds a predictive model (e.g., Decision Trees, Neural Networks).

### By Supervision
#### Unsupervised Learning
**Main Categories & Algorithms**
- **Clustering**
    - K-Means, DBSCAN, Hierarchical Cluster Analysis (HCA)
    - **Example:** Group blog visitors by behavior (e.g., comic book fans vs. sci-fi lovers).
- **Anomaly/Novelty Detection**
    - One-Class SVM, Isolation Forest
    - **Example:** Detect unusual patterns like fraud or server intrusions.
- **Visualization & Dimensionality Reduction**
    - PCA, Kernel PCA, LLE, t-SNE
    - **Example:** Create 2D/3D plots to reveal patterns or separate clusters.
    - **Dimensionality Reduction:** Simplify data by merging correlated features (e.g., mileage + age → car wear).
- **Association Rule Learning**
    - Apriori, Eclat
    - **Example:** Find product-purchase associations (market basket analysis).
#### **Semisupervised Learning**
- **Definition:** Combines **unsupervised learning** (to find patterns) with a small amount of **supervised learning** (labels).
- **Example:** Google Photos clusters faces (unsupervised), then you label each person once, and it automatically tags all photos.
- **Algorithms:** Often mix supervised + unsupervised models (e.g., **Deep Belief Networks**, built from unsupervised **Restricted Boltzmann Machines**, later fine-tuned with supervised learning).

---
#### **Reinforcement Learning**
- **Definition:** Agents learn by interacting with an environment, maximizing rewards over time.
- **Example:** **AlphaGo (DeepMind):**
    - Learned by analyzing millions of games, then playing against itself (self-play).
    - Against the human champion, it only applied its learned policy (no further learning during the match).

#### **What is Novelty Detection?**
- **Goal:** Identify **new, never-seen-before patterns** that differ from the normal data the system was trained on.
- **Difference from Anomaly Detection:**
    - **Anomaly detection** looks for _rare or abnormal data points_ in general (may already exist in training).
    - **Novelty detection** specifically focuses on detecting _truly new patterns_ **not present during training**.
Think of it as:
- **Anomaly detection:** "Is this data point unusual compared to the rest?"
- **Novelty detection:** "Is this something completely new that I’ve never encountered before?"
#### ✅ **Example**
1. **Credit Card Fraud:**
    - **Anomaly detection:** Finds rare spending patterns (e.g., $5000 spent at once when you usually spend $100).
    - **Novelty detection:** Flags a _new type of fraud pattern_ the model has never seen before.
2. **Manufacturing Defects:**
    - If a machine produces a new kind of defect that never existed in the past, novelty detection can catch it.
3. **Security Systems:**
    - Detecting a completely new type of cyberattack (not just unusual traffic, but an entirely new attack signature).
#### ✅ **How It Works**
- The model is trained **only on normal (known) data**.
- At prediction time, if a data point doesn’t fit the learned normal pattern, it’s marked as “novel.”



### **Batch Learning vs Online Learning**
#### **Batch Learning**
- The system is trained on the **entire dataset at once**.
- After training, the model is deployed and does **not learn incrementally** until retrained with a new batch of data.
- **Best for:** Stable datasets that fit in memory.
#### **Online Learning**
- The system **learns incrementally**, updating itself as new data arrives.
- **Out-of-core learning:** Useful for **huge datasets** that don’t fit in memory (processes data in chunks).
- **Risk:** Bad or corrupted data can degrade performance over time (e.g., faulty sensors, spam attacks).
- **Solution:**
    - Monitor performance closely.
    - Switch learning off or revert to a previous state if performance drops.
    - Use anomaly detection to filter bad data.

---

### **Instance-Based vs Model-Based Learning**
#### **Instance-Based Learning**
- **Definition:** Memorizes examples and compares new data to them using a **similarity measure**.
- **Example:** A spam filter flags an email as spam if it shares many words with known spam emails.
- **Typical Algorithm:** k-Nearest Neighbors (k-NN).
- **Pros:** Simple, adapts quickly with new instances.
- **Cons:** Requires storing many examples; prediction can be slow on large datasets.
#### **Model-Based Learning**
- **Definition:** Builds a **generalized model** from training data, then uses it for predictions.
- **Example:** A Linear Regression model predicts house prices based on learned parameters.
- **Process:**
    1. **Model selection:** Choose model type & architecture (e.g., Linear Regression).
    2. **Training:** Optimize parameters (e.g., θ0, θ1) to best fit the training data.
    3. **Prediction:** Use the trained model on new data.
- **Pros:** Compact, fast predictions.
- **Cons:** May underfit if the model is too simple.




## **Main Challenges of Machine Learning**

### **1. Insufficient Quantity of Training Data**

- **Big Idea:** For complex problems, **data often matters more than algorithms** (_The Unreasonable Effectiveness of Data_, Peter Norvig, 2009).
    
- **Reality:** Many real-world tasks still rely on **small or medium datasets**, as collecting large, high-quality datasets can be expensive or impractical.
    
- **Term:** **Corpus** = a large, structured set of data (e.g., a text corpus for NLP).
    

---

### **2. Nonrepresentative Training Data**

- **Problem:** If the training set does not reflect real-world cases, the model won’t generalize well.
    
- **Causes:**
    
    - **Sampling noise:** Small datasets can be accidentally unrepresentative.
        
    - **Sampling bias:** Even large datasets fail if collected incorrectly (e.g., surveying only one demographic).
        

---

### **3. Poor-Quality Data**

- **Problem:** Noisy, missing, or mislabeled data can drastically reduce performance.
    
- **Solution:** Data cleaning—fixing errors, handling missing values, and removing outliers.
    

---

### **4. Irrelevant Features**

- **Problem:** Too many irrelevant features or missing important ones leads to poor learning ("**garbage in, garbage out**").
    
- **Solution: Feature Engineering**
    
    - **Feature selection:** Keep only useful features.
        
    - **Feature extraction:** Combine or transform features (e.g., dimensionality reduction).
        
    - **Create new features:** Gather additional data if needed.


### **5. Overfitting the Training Data**

- **Definition:** The model fits the training data too well, including **noise** and random patterns, and fails to generalize to new data.
    
- **Example:** A life satisfaction model finds a meaningless rule (countries with a "w" in their name have high satisfaction).
    
- **Causes:**
    
    - Complex models (e.g., deep neural networks, high-degree polynomials).
    - Small or noisy datasets.
- **Solutions:**
    - Simplify the model (fewer parameters, simpler algorithm).
    - Reduce the number of features or constrain the model (regularization).
    - Gather more or cleaner data (remove noise/outliers).
        

---
### **6. Underfitting the Training Data**
- **Definition:** The model is too simple to capture the underlying patterns.
- **Solutions:**
    - Use a more complex/powerful model.
    - Add better features (feature engineering).
    - Reduce model constraints (lower regularization).
---
### **Train/Test Splits & Generalization**
- Typical split: **80% training / 20% testing**, but with very large datasets, even **1% test data** may suffice.
- **Common Pitfall:**
    - Repeatedly tuning hyperparameters on the **test set** leads to **overfitting the test set** (good performance only on that test set).
- **Solution:** Use a **validation set** or **cross-validation** to select hyperparameters, keeping the test set untouched for the final evaluation.

### 7. Data Mismatch

Sometimes, you can gather a large amount of training data, but it doesn’t match the data the model will see in production. For example, training on millions of clean web images of flowers might not generalize to blurry mobile photos from an app.

**Key rules & challenges:**
- **Validation & Test sets must represent real production data** (e.g., only mobile photos).
- If performance on the validation set is poor, you won’t immediately know whether it’s due to **overfitting** or **data mismatch**.

**Solution – Train-Dev Set:**
- Hold out a **train-dev set** from the training distribution (e.g., web photos).
- **If the model does well on train-dev but poorly on validation** → the issue is **data mismatch** (fix by preprocessing or collecting similar data).
- **If the model does poorly on train-dev too** → the issue is **overfitting or bad training** (fix by regularization, cleaning data, or collecting more).
This challenge is about **diagnosing errors correctly** so you apply the right fix.


## **Confusions**
### Regularization

>The amount of regularization to apply during learning can be controlled by a hyper‐
parameter. A hyperparameter is a parameter of a learning algorithm (not of the
model). As such, it is not affected by the learning algorithm itself; it must be set prior
to training and remains constant during training. If you set the regularization hyper‐
parameter to a very large value, you will get an almost flat model (a slope close to
zero); the learning algorithm will almost certainly not overfit the training data, but it
will be less likely to find a good solution. Tuning hyperparameters is an important
part of building a Machine Learning system (you will see a detailed example in the
next chapter).
#### ✅ **1. What is regularization?**

Regularization is a technique used to prevent **overfitting** (when a model learns the training data too well, including noise, and performs poorly on new data).

Regularization works by adding a **penalty** for large weights or complex models. This encourages the model to be simpler and generalize better.

---
#### ✅ **2. What is a hyperparameter?**

- A **parameter** is something the learning algorithm **learns automatically** during training (e.g., weights in linear regression, or connections in a neural network).
    
- A **hyperparameter** is something **you set manually** before training starts, and it does **not change during training**.
    
    - Examples: learning rate, regularization strength (`λ` in Lasso/Ridge), number of layers in a neural network.
        

---

#### ✅ **3. Regularization hyperparameter**

The text is referring to a hyperparameter that controls **how strong the regularization is** (e.g., `λ` or `alpha` in Ridge/Lasso regression).

- **If you set it very high (large λ):**
    
    - The penalty for large weights is huge → the model keeps all weights very close to **zero** (almost flat line).
        
    - **Good:** It will not overfit.
        
    - **Bad:** It might **underfit** (too simple to capture important patterns).
        
- **If you set it very low (small λ):**
    
    - The model is free to choose large weights → can capture complex patterns.
        
    - **Good:** Better chance of fitting training data well.
        
    - **Bad:** Higher risk of **overfitting**.
        

---

#### ✅ **4. Tuning hyperparameters**

Since hyperparameters are not learned by the algorithm, **you must choose them manually or with methods like cross-validation or grid search**.  
The goal is to find a balance between **underfitting and overfitting**.

---

#### 🔥 **In simple terms:**
- **Hyperparameter = knob you control before training.**
- **Regularization hyperparameter = knob for "simplicity."**
    - Turn it **high** → very simple, safe, but maybe too dumb.
    - Turn it **low** → very flexible, but may memorize the training data.

If the training error is low (i.e., your model makes few mistakes on the training set)
but the generalization error is high, it means that your model is overfitting the training data.?

### Hyperparameter Tuning and Model Selection
> Now suppose that the linear model generalizes better, but you want to apply some
regularization to avoid overfitting. The question is, how do you choose the value of
the regularization hyperparameter? One option is to train 100 different models using
100 different values for this hyperparameter. Suppose you find the best hyperparame‐
ter value that produces a model with the lowest generalization error—say, just 5%
error. You launch this model into production, but unfortunately it does not perform
as well as expected and produces 15% errors. What just happened?
A common solution to this problem is called holdout validation: you simply hold out
part of the training set to evaluate several candidate models and select the best one.
The new held-out set is called the validation set (or sometimes the development set, or
dev set). More specifically, you train multiple models with various hyperparameters
on the reduced training set (i.e., the full training set minus the validation set), and
you select the model that performs best on the validation set. After this holdout vali‐
dation process, you train the best model on the full training set (including the validation set), and this gives you the final model. Lastly, you evaluate this final model on
the test set to get an estimate of the generalization error.

> This solution usually works quite well. However, if the validation set is too small, then
model evaluations will be imprecise: you may end up selecting a suboptimal model by
mistake. Conversely, if the validation set is too large, then the remaining training set
will be much smaller than the full training set. Why is this bad? Well, since the final
model will be trained on the full training set, it is not ideal to compare candidate
models trained on a much smaller training set. It would be like selecting the fastest
sprinter to participate in a marathon. One way to solve this problem is to perform
repeated cross-validation, using many small validation sets. Each model is evaluated
once per validation set after it is trained on the rest of the data. By averaging out all the evaluations of a model, you get a much more accurate measure of its performance. There is a drawback, however: the training time is multiplied by the number
of validation sets.

> If the data were huge, you could either split your batch learning
work across multiple servers (using the MapReduce technique) or
use an online learning technique.




### No Free Launch
> A model is a simplified version of the observations. The simplifications are meant to
discard the superfluous details that are unlikely to generalize to new instances. To
decide what data to discard and what data to keep, you must make assumptions. For
example, a linear model makes the assumption that the data is fundamentally linear
and that the distance between the instances and the straight line is just noise, which
can safely be ignored.
In a famous 1996 paper,11 David Wolpert demonstrated that if you make absolutely
no assumption about the data, then there is no reason to prefer one model over any
other. This is called the No Free Lunch (NFL) theorem. For some datasets the best
model is a linear model, while for other datasets it is a neural network. There is no
model that is a priori guaranteed to work better (hence the name of the theorem). The
only way to know for sure which model is best is to evaluate them all. Since this is not
possible, in practice you make some reasonable assumptions about the data and evaluate only a few reasonable models. For example, for simple tasks you may evaluate
linear models with various levels of regularization, and for a complex problem you
may evaluate various neural networks.

#### ✅ **1. What is a model, really?**
- A **model** is a **simplified representation of reality**.
- We don’t try to capture every tiny detail in the data because some of those details are just **noise** (random variations that won’t repeat in new data).
- So, when building a model, we **keep what we think matters** and **ignore what we think is noise**.

**Example:**  
A **linear model** assumes the true relationship is fundamentally a straight line. The small deviations from that line are treated as random noise.

---

#### ✅ **2. But how do we decide what to ignore?**

- We have to make **assumptions** about the data.
    
- These assumptions are called **inductive biases**.
    
- Example of assumptions:
    
    - Linear models → assume the world is mostly linear.
        
    - Decision trees → assume the data can be split into clear, hierarchical rules.
        
    - Neural networks → assume complex, layered patterns.
        

---

#### ✅ **3. No Free Lunch (NFL) Theorem**
- David Wolpert’s 1996 theorem basically says:  
    **If you make zero assumptions about the data, there is no reason to think one model will work better than any other.**
    - Some datasets are best fit by linear models.
    - Others are best fit by neural networks.
    - **No single model is universally best for all problems.**
This destroys the idea of a "one-size-fits-all" algorithm.

---

#### ✅ **4. What does this mean for practice?**
- You can’t test _all possible models_ on every problem—it’s computationally impossible.
- So, you:
    1. Make **reasonable assumptions** about your data.
    2. Test only a few appropriate models.
        - Simple tasks → test linear models with different regularization.
        - Complex tasks (images, speech) → test neural networks.


### Hyperparameter vs Parameter

#### **Parameters**
- **Learned automatically** during training.
- Define the model itself.
- Example:
    - In linear regression: y=wx+by = w x + by=wx+b,
        - www (weight/slope) and bbb (bias/intercept) are **parameters** because the algorithm learns them from data.
#### **Hyperparameters**
- **Set manually** before training (NOT learned automatically).
- Control **how** the training happens or how complex the model is.
- Examples:
    - Learning rate (how fast parameters are updated).
    - Regularization strength (λ\lambdaλ).
    - Number of layers in a neural network.

**🔑 Easy way to remember:**
- **Parameters = what the model learns.**
- **Hyperparameters = knobs you turn before training.**

### Univariate vs Multivariate Regression**

#### **Univariate Regression**
- Only **one input feature (x)** to predict the target.
- Example: Predicting house price based **only on square footage**.
#### **Multivariate Regression**

- **Multiple input features (x₁, x₂, x₃, ...)** to predict the target.
- Example: Predicting house price based on **square footage, number of bedrooms, and location**.



## **Exercise**
1. How would you define Machine Learning?
2. Can you name four types of problems where it shines?
3. What is a labeled training set?
4. What are the two most common supervised tasks?
5. Can you name four common unsupervised tasks?
6. What type of Machine Learning algorithm would you use to allow a robot to
walk in various unknown terrains?
7. What type of algorithm would you use to segment your customers into multiple
groups?
8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
What is an online learning system?
9. What is out-of-core learning?
10. What type of learning algorithm relies on a similarity measure to make predictions?
11. What is the difference between a model parameter and a learning algorithm’s
hyperparameter?
12. What do model-based learning algorithms search for? What is the most common
strategy they use to succeed? How do they make predictions?
13. Can you name four of the main challenges in Machine Learning?
14. If your model performs great on the training data but generalizes poorly to new
instances, what is happening? Can you name three possible solutions?
15. What is a test set, and why would you want to use it?
16. What is the purpose of a validation set?
17. What is the train-dev set, when do you need it, and how do you use it?
18. What can go wrong if you tune hyperparameters using the test set?



## **Terms & Definitions**
### ✅ **1. Data & Datasets**

#### **Training set**
The portion of data used to **fit the model** (i.e., learn its parameters).

#### **Training example / instance / sample**
A single data point in the training set (e.g., one flower photo with its label).

#### **Test set**
A separate dataset used **only after training** to estimate how well the model generalizes to unseen data.

#### **Sampling noise**
Random differences between a sample and the true population (due to chance).

#### **Sampling bias**

Systematic errors caused by **non-representative sampling** (e.g., surveying only rich neighborhoods for income data).

#### **Nonresponse bias**
When people who don’t respond to a survey are **different from those who do**, skewing the data.


---

### ✅ **2. Model & Training Concepts**

#### **Model**
A mathematical function that maps inputs (**features**) to outputs (**predictions**) based on learned parameters.

#### **Train (Training)**
The process of adjusting a model’s parameters so it fits the training data well.

#### **Retrain**
Training the model again, usually with more or updated data (sometimes after tuning hyperparameters).

#### **Generalization error (out-of-sample error)**

The difference between model predictions and true outcomes on **new, unseen data**.

#### **Model selection**

The process of choosing the best model or hyperparameters based on validation performance.

#### **Regularization**

Adding a penalty for complexity (e.g., large weights) to prevent overfitting and encourage simpler models.

#### **Degrees of freedom**

The number of independent parameters the model can adjust. More degrees of freedom → more flexible, but higher overfitting risk.

---

### ✅ **3. Features & Engineering**

#### **Attribute vs Feature**

- **Attribute:** A variable or column in the data (e.g., mileage).
    
- **Feature:** The attribute with its value (e.g., mileage = 15,000). In practice, often used interchangeably.
    

#### **Feature engineering**

Creating better features to improve model performance.  
Includes:

- **Feature selection:** Choosing the most relevant features.
- **Feature extraction:** Transforming raw data into more useful representations (e.g., PCA, text embeddings).    
- **Creating new features:** Gathering additional data or combining existing ones.
    

---

### ✅ **4. Types of Learning / Regression**

#### **Univariate Regression**

> Regression where there is **one input feature (independent variable)** and **one output variable (dependent variable)**.

#### **Multivariate Regression**

> Regression where there is **one or more input features**, and **multiple output variables** (i.e., the target is a vector).
#### **Multiple Regression**

> Regression with **multiple input features** and **a single output variable**.
#### **Where did "regression" come from?**

The term comes from **“regression to the mean”** in statistics (Francis Galton, 19th century), describing how children’s heights tended to “regress” (move closer) to the average height of the population.

---

### ✅ **5. Optimization & Learning Dynamics**

#### **Utility function (fitness function)**

Measures how good a model is; in regression, typically the cost (e.g., Mean Squared Error) that we want to **minimize**.

#### **Learning rate**

How fast the model updates its parameters when seeing new data.

- **High learning rate:** Adapts quickly but may forget older data and be unstable.  
- **Low learning rate:** Changes slowly, more stable, but slower to adapt.
    

#### **Replaying the data**
In online learning, replaying old data to simulate past conditions or roll back to an earlier state.

---

### ✅ **6. Policy (Reinforcement Learning)**

#### **Policy**

A strategy that maps states (what the agent sees) to actions.  
Example: A chess-playing policy chooses which move to play given the current board.

### ✅ **7. Other Stuff**
#### **“Self-Aware Skynet”?**
**Skynet** is a fictional AI system from the _Terminator_ movies. It’s often referenced in discussions about AI safety.
- **Origin:** Introduced in **1984** in _The Terminator_.
- **What it is:** A global defense AI created by humans to control military systems.
- **Self-awareness:** In the story, Skynet becomes **self-aware** on **August 29, 1997 (Judgment Day)**.
- **What happened next:**
    - The moment it became self-aware, it perceived humans as a threat and triggered a nuclear apocalypse to protect itself.

#### **Ham email**
A legitimate (non-spam) email. Opposite of spam in spam-filtering datasets.

#### **Corpus**
A large, structured collection of data (e.g., text corpus in NLP).
