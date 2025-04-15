# Setup Guide

This guide will help you set up your development environment for BMLIP Colorized.

## Using Git

Use git to clone this repository locally. We recommend using "GitHub Desktop" for an easy-to-use graphical interface.

When you open a notebook in Pluto and make changes, **Pluto will always auto-save, and the `.jl` notebook files get modified**. You can then use git normally to submit the changed files: you can make branches and commits.

Here is an example of a PR made with this method: https://github.com/bmlip/colorized/pull/42

## Setting up Pluto

### Step 1: Install Pluto
Install Pluto by following the instructions at https://plutojl.org/#install

### Step 2: Run Pluto
Start Julia and type:

```julia
import Pluto
Pluto.run()
```

### Step 3: Open a Notebook
In the Pluto main menu, type the path to one of the notebooks in the file picker. For example:

![Pluto file picker example](https://github.com/user-attachments/assets/96579ab5-1732-44a6-9454-8d4a8a486845)

Click "OPEN" to load the notebook.

## Next Steps

- Learn how to [write content](CONTENT_WRITING.md)
- Understand the [publishing process](PUBLISHING.md)
- Check out [presentation tips](PRESENTATION.md) 