# BMLIP *2025-2026 edition*

This is the **work-in-pogress** *new* version of the course [Bayesian Machine Learning and Information Processing](https://github.com/bertdv/BMLIP). ✨

There will be new lecture materials based on [Pluto.jl](https://plutojl.org/) with interactivity and color!


# How to preview notebooks
This repository has a GitHub Action that automatically generates a static website on every commit. This uses the [Pluto static export system](https://plutojl.org/en/docs/notebooks-online/) to generate static HTML files.

You can see an index of all exported files at the repository website (link in top-right).

✅ Recent commit? For progress and status of the site generation system, check out the [**Actions** tab](https://github.com/biaslab/BMLIP-colorized/actions).


## Notebook from Pull Request
To preview a notebook from a Pull Request, open it locally. The quickest way to do so:

### Step 1
Open the PR and go to the "Files" tab. Find the notebook file that you want to preview.

### Step 2
Click the (...) button and **Right Click** the *View raw* option. Click **Copy link URL**.

<img width="1403" alt="Scherm­afbeelding 2025-01-14 om 18 02 22" src="https://github.com/user-attachments/assets/6fc1011b-8a8d-4419-b97c-a972b779a950" />

### Step 3
Open Pluto (`import Pluto; Pluto.run()`). In the main menu, **paste the URL** in the file picker and press **Open**.

<img width="1011" alt="image" src="https://github.com/user-attachments/assets/be98b029-41dc-4a8d-8c03-31730fb9a2bd" />


# How to run notebooks locally

## Method A: clone the repository
You can clone the repository. When you have the notebook files (`.jl`) on your computer, you can run them with Pluto!

### Step 1
Install Pluto! Check out https://plutojl.org/#install for instructions.

### Step 2
Run Pluto! Start Julia and type:

```julia
import Pluto
Pluto.run()
```

### Step 3
In the Pluto main menu, type the path to one of the notebooks in the file picker. For example:

<img width="1161" alt="image" src="https://github.com/user-attachments/assets/96579ab5-1732-44a6-9454-8d4a8a486845" />


## Method B: open a single notebook
If you don't want to clone the complete repository, you can also run a single notebook directly from its URL.

### Step 1 & 2
Open and run Pluto, see Method A.

### Step 3
Find a notebook file (`.jl`) in this repostory that you want to run. Copy the URL, e.g.

```
https://github.com/biaslab/BMLIP-colorized/blob/main/lectures/B12%20Intelligent%20Agents%20and%20Active%20Inference.jl
```

*Tip: You don't need the "raw URL", any type of URL will work*

### Step 4
In the Pluto's main menu, **paste the URL** in the file picker and press **Open**.

<img width="1011" alt="image" src="https://github.com/user-attachments/assets/be98b029-41dc-4a8d-8c03-31730fb9a2bd" />






