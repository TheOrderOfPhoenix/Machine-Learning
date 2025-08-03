# Python Basics

Includes *Virtual Env*, *Collab* and link to *NumPy* tutorials

## Virtual Environment

Setting up a virtual environment is a smart way to keep your project clean and reproducible—especially when you're working with packages like NumPy and Seaborn in Jupyter notebooks. Here's a **step-by-step guide** using both **Conda** and **venv**, so you can choose whichever suits your workflow best:

### ✅ Option 1: Using **Conda** (Recommended for Data Science projects)

Conda is ideal if you already use Anaconda or Miniconda. It handles scientific packages very well.

#### 🔧 Step-by-Step

1. **Navigate to your project folder** in terminal:

   ```bash
   cd path/to/your/project
   ```

2. **Create a new conda environment**:

   ```bash
   conda create -n myenv python=3.11
   ```

   Replace `myenv` with your preferred environment name.

3. **Activate the environment**:

   ```bash
   conda activate myenv
   ```

4. **Install packages**:

   ```bash
   conda install numpy seaborn jupyter
   ```

5. (Optional but useful) **Install `ipykernel` so Jupyter can use this environment**:

   ```bash
   conda install ipykernel
   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
   ```

6. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

7. In the notebook interface, select the kernel: `Kernel` > `Change Kernel` > `Python (myenv)`

### ✅ Option 2: Using **`venv`** + pip (More lightweight, universal)

If you're not using Conda and want something lighter:

#### 🔧 Step-by-Step

1. **Navigate to your project folder**:

   ```bash
   cd path/to/your/project
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   ```

3. **Activate the environment**:

   * On Windows:

     ```bash
     venv\Scripts\activate
     ```
   * On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install your packages**:

   ```bash
   pip install numpy seaborn notebook ipykernel
   ```

5. **Add the environment as a Jupyter kernel**:

   ```bash
   python -m ipykernel install --user --name=venv --display-name "Python (venv)"
   ```

6. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

### 📦 Bonus: Save and Share Dependencies

At any point, save your environment to a file:

* **Conda**:

  ```bash
  conda env export > environment.yml
  ```

* **venv/pip**:

  ```bash
  pip freeze > requirements.txt
  ```

This makes it super easy to recreate the same environment later with:

* `conda env create -f environment.yml`
* `pip install -r requirements.txt`

## Git

Cloning your project with Git is a solid move—especially when you're working with notebooks, dependencies, and scientific experimentation. 
Let's talk **best practices** for setting up, cloning, and maintaining a clean, shareable, and reproducible data science project (like your NumPy + Seaborn + Jupyter one).

### ✅ Best Practices for Cloning and Managing a Jupyter-Based Data Science Project with Git

#### 📁 1. **Project Folder Structure**

Organize your repo for clarity and scale. Here's a clean baseline structure:

```
your-project/
│
├── notebooks/          # Jupyter notebooks (analysis, experiments)
│   ├── analysis.ipynb
│   └── viz.ipynb
│
├── data/               # Raw or processed datasets (often .gitignored)
│   └── ...
│
├── env/ or environment.yml  # Conda env (or requirements.txt for pip/venv)
│
├── .gitignore
├── README.md
├── LICENSE (optional)
└── setup.sh or setup_instructions.md (optional script/instructions to set up env)
```

#### 🧠 2. **Cloning the Project**

From any machine:

```bash
git clone https://github.com/your-username/your-project.git
cd your-project
```

#### 🛠️ 3. **Environment Setup (Best Practice)**

As soon as you clone it, you should recreate the environment:

##### a) If using Conda:

```bash
conda env create -f environment.yml
conda activate your-env-name
jupyter notebook
```

##### b) If using `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook
```

#### 🧾 4. **Git Best Practices for Jupyter Projects**

##### ✅ Track only what you should:

* **Use `.gitignore` wisely**:
  Example `.gitignore`:

  ```gitignore
  __pycache__/
  .ipynb_checkpoints/
  .DS_Store
  venv/
  env/
  data/
  *.pyc
  ```

* **Avoid committing datasets or output files unless small and necessary.**

##### ✅ Commit clear, atomic changes:

Avoid "big dump" commits. Instead, do things like:

```bash
git add notebooks/viz.ipynb
git commit -m "Add correlation heatmap using seaborn"
```

##### ✅ Use branches if you're experimenting:

```bash
git checkout -b try-different-plot-style
```

#### 📚 5. **Include a Good `README.md`**

People (including future you!) will appreciate context. Your README should include:

````markdown
# My Data Project

This project explores [topic] using NumPy and Seaborn in Jupyter notebooks.

## Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project
````

2. Create the environment:

   ```bash
   conda env create -f environment.yml
   conda activate myenv
   ```

3. Launch notebooks:

   ```bash
   jupyter notebook
   ```

### Notebooks

* `notebooks/analysis.ipynb`: Data exploration
* `notebooks/viz.ipynb`: Visualizations with seaborn

### Requirements

* Python 3.11
* NumPy
* Seaborn
* Jupyter

````

### 🧪 Bonus: Track Notebook Diffs Like a Pro

By default, Git tracks notebooks as JSON blobs. To improve this:
- Use [nbdime](https://github.com/jupyter/nbdime):
  ```bash
  pip install nbdime
  nbdime config-git --enable
````

This gives you readable diffs like `git diff notebooks/plot.ipynb`.


### 🏁 Final Thoughts

✅ **Use virtual environments** for reproducibility
✅ **Keep data out of the repo**, unless it's tiny
✅ **Write helpful commits and README**
✅ **Structure your repo like you're building a product, not a playground**

## Google Collab

Working with **Google Colab** as a data scientist can be super powerful if you use it wisely. 
It's more than just a free Jupyter notebook in the cloud — with the right techniques, it can feel almost like a full-blown data science IDE. 
Here are **pro tips and hacks** to level up your workflow:

### 🧠 1. **Environment Setup Like a Pro**

* **Use a requirements.txt**
  At the top of your notebook, run:

  ```python
  !pip install -r requirements.txt
  ```

  This keeps your environment reproducible. Store this file in your repo.

* **Use virtual environments (indirectly)**
  While you can't create Conda envs in Colab easily, you can simulate isolated environments using `pip install` and `%pip` or `%conda` in Jupyter magics (though `%conda` needs some tricks).

### ⚡ 2. **Speed & Resource Boosting**

* **Get free GPU/TPU**

  * Go to `Runtime > Change runtime type > Hardware accelerator > GPU/TPU`.
  * For deep learning or large matrix ops, even a **Tesla T4** gives a huge boost.

* **Reconnect to prevent timeouts**
  Use this in your browser console to auto-click “Reconnect”:

  ```javascript
  function ClickConnect(){
    console.log("Auto reconnecting...");
    document.querySelector("colab-connect-button").shadowRoot.querySelector("#connect").click();
  }
  setInterval(ClickConnect, 60000);
  ```

### 🧩 3. **Mounting Google Drive = Easy Data Access**

* Store large datasets, models, or checkpoints on Google Drive:

  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

* Use relative paths like:

  ```python
  data_path = "/content/drive/MyDrive/my_project/data.csv"
  ```

### 📊 4. **Great Visualizations, Fast**

* Use `%matplotlib inline`, `seaborn`, `plotly`, or `altair` for interactive/beautiful charts.
* If you're doing ML, use `yellowbrick` or `mlxtend` for quick diagnostic visuals.

### 💾 5. **Persist Your Models & Data**

* Save models to Drive:

  ```python
  import joblib
  joblib.dump(model, '/content/drive/MyDrive/model.pkl')
  ```

* Save plots/images:

  ```python
  plt.savefig('/content/drive/MyDrive/plot.png')
  ```

### 🚀 6. **Magics to Save Time**

* `%time` / `%timeit` – Performance insights
* `%load_ext autoreload` – Auto-reloads modules
* `%debug` – Drops into interactive debugger on exception
* `%who` – Lists variables in memory
* `%history` – Shows command history

### 🤝 7. **Collaboration Superpowers**

* **Comment & Share:** Use the share link like Google Docs

* **Version control?** Sync with GitHub:

  ```python
  !git clone https://github.com/yourname/yourrepo.git
  ```

* You can even **push back** changes:

  ```python
  !git add . && git commit -m "update" && git push
  ```

### 🛠 8. **Install Anything, Do Anything**

* Need external tools?

  ```python
  !apt install ffmpeg
  !wget https://some-url.com/file.zip
  ```

* Even use `ngrok` or `flask` to expose a local web app:

  ```python
  !pip install flask-ngrok
  ```

### 📁 9. **Organize Code With Scripts**

* Instead of cluttering your notebook:

  * Store functions in `.py` scripts and import them:

    ```python
    from my_utils import preprocess_data
    ```

* Reimport updated files:

  ```python
  import importlib
  importlib.reload(my_utils)
  ```

### 🔒 10. **Security Warning: Don’t Share Tokens in Code**

If you're using APIs, load keys like this:

```python
import os
os.environ['API_KEY'] = "your_token"
```

Then access it in code:

```python
key = os.environ['API_KEY']
```

### ⚙️ Bonus: Use Extensions!

Colab doesn't support full Jupyter extensions, but there are *hacks*:

* Use custom CSS with JavaScript
* Or, for advanced notebooks, try **JupyterLab** on platforms like [Kaggle Notebooks](https://www.kaggle.com/code) or [Deepnote](https://deepnote.com/) if Colab becomes limiting.

## NumPy

Check this [Link](https://github.com/AminGh05/NumPyTut) to find some notebooks to learn more about **NumPy**
