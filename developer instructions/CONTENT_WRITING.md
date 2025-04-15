# Content Writing Guide

This guide covers how to write and modify lectures in BMLIP Colorized.

## Markdown Basics

We use Markdown for prose content. Add a new cell, and write:

```julia
md"""
Some **content**.
"""
```

> [!TIP]
> You can use the keyboard shortcut **`Ctrl + M`** to quickly add/remove the `md"""` literal.

## Content Structure

### Blocks, Headers, and Quotes
There are many Markdown features to create callouts, code blocks, quotes, lists, etc. The `md"""` macro is from the Julia stdlib Markdown.jl. Read the full list of features here: https://docs.julialang.org/en/v1/stdlib/Markdown/

### Useful Widgets

**PlutoUI.jl** provides:
- `@bind` widgets like sliders
- `details` for collapsible sections ("click to read more")

**PlutoTeachingTools.jl** provides:
- `aside` for placing content in the side margin

## Package Management

Pluto has a built-in package manager that is automatically reproducible. With `using` or `import` you can import any package you want, Pluto will take care of the rest.

Read more: [https://plutojl.org/en/docs/packages/](https://plutojl.org/en/docs/packages/)

## LaTeX and Math

You can use LaTeX in Markdown! Because of conflicts with interpolation, we recommend using backticks:

```julia
md"""
Here is some ``\srt{inline}`` math.

```math
This = \frac{block}{math}
```
"""
```

You can also use `\begin{align}` etc inside a math block.

## Adding Images

The best method is to use GitHub's Issues feature:

1. Open a new issue in any public GitHub repository
2. Drag your image into the writing box
3. Copy the generated image URL
4. Use it in your Markdown:

```julia
md"""
My image:

![some description](https://github.com/user-attachments/assets/your-image-id)
"""
```

5. Close the issue (no need to post it)

## HTML in Notebooks

Pluto has first-class support for HTML and JavaScript. You can use HTML cells:

```julia
html"""
<h1>My header</h1>
<p style='color: red;'>This is a paragraph in red.</p>
"""
```

Or use HTML inside Markdown with interpolation:

```julia
md"""
My header:

$(html"<h1>My header</h1>")
"""
```

## Linking

### Between Lectures
Go to the course website, find the lecture you want to link, and use that URL.

### Within Lectures
Use HTML spans with IDs for linking:

```julia
md"""
# $(html"<span id=my-header>") My header
This is a paragraph.
"""
```

## Code in Pluto

Pluto has a stricter runtime than Jupyter to ensure reproducibility. Read more about the differences in [this article](https://featured.plutojl.org/basic/pluto%20for%20jupyter%20users).

## Next Steps

- Learn about the [publishing process](PUBLISHING.md)
- Check out [presentation tips](PRESENTATION.md) 