
> [!INFO]
> This document: How to write new lectures, how to modify lectures.



# Setup

## Using git
Use git (Fons recommends "GitHub Desktop") to clone this repository locally.

When you open a notebook in Pluto (see next chapter) and make changes, **Pluto will always auto-save, and the `.jl` notebook files get modified**. You can then use git normally to submit the changed files: you can make branches and commits.

Here is an example of a PR made with this method: https://github.com/biaslab/BMLIP-colorized/pull/42

## Using Pluto
First, set up Pluto: https://plutojl.org/#install


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

Click "OPEN".

# Publishing
To submit content, **make a PR**. This allows our automatic testing system to check for possible errors.

When you merge the PR, this creates a **commit on main**. This repository has a GitHub Action that automatically generates a static website on every commit. This uses the [Pluto static export system](https://plutojl.org/en/docs/notebooks-online/) to generate static HTML files. **When the action is done, the website will show your new notebook.**

The PlutoSliderServer is also updated automatically on every commit.

# Writing content


## Markdown
We use Markdown for prose content. Add a new cell, and write:

```julia
md"""
Some **content**.
"""
```

> [!TIP]
> You can use the keyboard shortcut **`Ctrl + M`** to quickly add/remove the `md"""` literal.

Markdown is not a complete standard: every Markdown environment (Julia's stdlib, GitHub's renderer, Jupyter, dynalist, etc) is **different** and has different features.

## Structure: blocks, headers, quotes
There are many Markdown features to create a callout, code block, quote, list, etc. The `md"""` macro is from the Julia stdlib Markdown.jl (not from Pluto!), and you can read the full list of features here: https://docs.julialang.org/en/v1/stdlib/Markdown/

Also very useful is are widgets from two packages:

**PlutoUI.jl** is mostly for `@bind` widgets like sliders, but it also has:
- `details` for a collapsable section ("click to read more")

**PlutoTeachingTools.jl**
- `aside` for placing content in the side margin


## Packages, Project.toml, Manifest.toml
Pluto has a built-in package manager that is automatically reproducible. With `using` or `import` you can import any package you want, Pluto will take care of the rest :)

Read more: [https://plutojl.org/en/docs/packages/](https://plutojl.org/en/docs/packages/)




## LaTeX
You can use LaTeX in Markdown! Different syntax is possible, including `$math$` like in LaTeX text.

But because of conflicts with interpolation, Fons recommend using backticks:

`````julia
md"""
Here is some ``\srt{inline}`` math.

```math
This = \frac{block}{math}
```

"""
`````

You can also use `\begin{align}` etc inside a math block.


Two notable differences between Jupyter's markdown and Julia stdlib:

## Interpolation
Julia Markdown supports **interpolation** with `$`. This lets you insert numbers, data, plots inside your Markdown prose.


```julia
N = size(X, 2)
```

```julia
md"""
Our dataset contains $(N) observations.
"""
```

#### HTML inside Markdown
Jupyter supports HTML inside Markdown. Julia does not. A workaround is to iterpolate an HTML snippet:

```julia
md"""
In the above figure, the Gaussian components are shown in $(html"<span style='color: red'>red</span>") and the pdf of the mixture models in $(html"<span style='color: blue'>blue</span>").
"""
```



## Adding images
In Jupyter, you often load images using a relative path, e.g.

```html
<!-- ⛔️ do not use in Pluto -->
<img src="./figures/something-cool.svg">
```

In Pluto, we **don't recommend this**. Instead, use images from the web, using their URL. 

You can link to images on github:

```julia
# ⚠️ maybe use in Pluto, but be careful to use a version-pinned URL
md"""
![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/figures/Bishop-Figure4.5b.png?raw=true)
"""
```

But the best method is to **(ab)use GitHub's Issues feature**! This gives the most reliable result and it is easy to do!

### Step 1: Open a new issue anywhere
Go to a public GitHub repository and click "New issue".


### Step 2: Upload your image to get a URL
Drag your image into the writing box. It will upload and you get an image.


### Step 3: Copy the image URL
After uploading, an image tag is inserted. It will look like:

```
<img width="130" alt="Image" src="https://github.com/user-attachments/assets/e9dcc842-ada4-4508-bf53-bbe3c9f52a30" />
```

Here, **copy the URL part**, i.e.

```
https://github.com/user-attachments/assets/e9dcc842-ada4-4508-bf53-bbe3c9f52a30
```

### Step 4: Markdown
Now this is your image URL! You can use it in Pluto in a Markdown cell:

```julia
# ✅ use this in Pluto!
md"""
My image:

![some description](https://github.com/user-attachments/assets/e9dcc842-ada4-4508-bf53-bbe3c9f52a30)
"""
```

### Step 5: Cancel the issue
That's it! You can close the Issue tab, you do not need to post the issue.


## HTML in notebooks
Most content will be written in Markdown, but it is also possible (and encouraged!) to use HTML in Pluto! 

Pluto has first-class support for HTML and JavaScript, just like Markdown. But HTML is more powerful!


### HTML cells
The easiest way to use HTML is to create "html cells", using the Julia built-in `html"` string marcro:

```julia
html"""
<h1>My header</h1>
<p style='color: red;'>This is a paragraph in red.</p>


<!-- And you can use JavaScript! -->

<script>
alert("Hello from JS!")
</script>
"""
```

### HTML in Markdown
You can also use HTML inside Markdown cells. But remember, **Julia Markdown** does not support HTML directly (like Jupyter Markdown), so you need to use the `html"..."` macro together with interpolation `$(...)`:

```julia
md"""
My header:

$(html"<h1>My header</h1>")
"""
```

> [^WARNING]
> The Markdown stdlib can be **unpredictable with interpolation**, brackets and quotes. (It's a long story...) Try to use the single quote `'` inside HTML.
> 
> When things get too complex, define the interpolated object in another cell, and assign it to a variable. Then just interpolate the variable. E.g.
> ```julia
> thing = html"..."
> md"""Here is $(thing)"""
> ```


## Linking between lectures
Go to the course website, find the lecture you want to link, and use this as a URL.

## Linking within lectures
This is more difficult, because of missing functionality in the Julia Markdown stdlib.

### How does linking work on the web?
A URL can have a **fragment** at the end, after the `#`. This looks like:

```
https://somedomain.com/some/sub/path.html#THEFRAGMENT
```

The fragment is special, it will ask the WWW for this page:

```
https://somedomain.com/some/sub/path.html
```

And on the local side, the browser will load the page, and then navigate to the first HTML element with `id` set to `THEFRAGMENT`.

### How do I set an `id`?
It is not possible with the Julia Markdown stdlib to set an `id`. ☹️ So we need a special trick! This is what I recommend:

```julia
md"""
# $(html"<span id=my-header>") My header
This is a paragraph.
"""
```

This creates an HTML `span` (the *neutral* element) with the `id` set. Here we use 
- the `html"..."` string macro. This creates an `HTML` object that will render as 
- **interpolation** with `$(...)` to place the `HTML` object inside the header 



## Code in Pluto
When coming from Jupyter, you will notice that Pluto has a **stricter runtime**. These restrictions are there to help you write **reproducible**, **reactive**, clean and readable notebooks.

Pluto's restrictions will be frustrating at first (when coming from Jupyter), but Fons thinks that this is a small price to pay for reproducible, interactive notebooks!

### 👉 Read [this article](https://featured.plutojl.org/basic/pluto%20for%20jupyter%20users) about the differences between Jupyter and Pluto.




# Presenting in lectures

👉 You can use Pluto's **presentation mode**: https://plutojl.org/en/docs/presentation/