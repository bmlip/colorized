### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 0cfd4bc0-d294-11ef-3537-630954a9dd27
md"""
# 5SSD0 Course Syllabus

"""

# ╔═╡ 0cffef7e-d294-11ef-3dd5-1fd862260b70
md"""
## Learning Goals

This course provides an introduction to Bayesian machine learning and information processing systems. The Bayesian approach affords a unified and consistent treatment of many useful information processing systems. 

Upon successful completion of the course, students should be able to:

  * understand the essence of the Bayesian approach to information processing.
  * specify a solution to an information processing problem as a Bayesian inference task on a probabilistic model.
  * design a probabilistic model by a specifying a likelihood function and prior distribution;
  * Code the solution in a probabilistic programming package.
  * execute the Bayesian inference task either analytically or approximately.
  * evaluate the resulting solution by examination of Bayesian evidence.
  * be aware of the properties of commonly used probability distribitions such as the Gaussian, Gamma and multinomial distribution; models such as hidden Markov models and Gaussian mixture models; and inference methods such as the Laplace approximation, variational Bayes and message passing in a factor graph.

"""

# ╔═╡ 0d013750-d294-11ef-333c-d9eb7578fab2
md"""
## Entrance Requirements (pre-knowledge)

Undergraduate courses in Linear Algebra and Probability Theory (or Statistics). 

Some scientific programming experience, eg in MATLAB or Python. In this class, we use the [Julia](https://julialang.org/) programming language, which has a similar syntax to MATLAB, but is (close to) as fast as C. 

"""

# ╔═╡ 0d0142b6-d294-11ef-0297-e5bb923ad942
md"""
## Important Links

Please bookmark the following three websites:

1. The course homepage [http://bmlip.nl](https://biaslab.github.io/teaching/bmlip/) (or try [https://biaslab.github.io/teaching/bmlip](https://biaslab.github.io/teaching/bmlip/) ) contains links to all materials such as lecture notes and video lectures.
2. The [Piazza course site](https://piazza.com/tue.nl/winter2025/5ssd0/home) will be used for Q&A and communication.
3. The [Canvas course site](https://canvas.tue.nl/courses/30024) will be sparingly used for communication (mostly by ESA staff)

"""

# ╔═╡ 0d015ab4-d294-11ef-2e53-5339062c435c
md"""
## Materials

All materials can be accessed from the [course homepage](https://biaslab.github.io/teaching/bmlip).

Materials consist of the following resources:

  * Mandatory

      * Lecture notes
      * Probabilistic Programming (PP) notes
      * The lecture notes and probabilistic programming notes contain the mandatory materials. Some lecture notes are extended by a reading assignment, see the first cell in the lecture notes. These reading assignment are also part of the mandatory materials.
  * Optional materials to help understand the lecture and PP notes

      * video recordings of the Q2-2023 lecture series
      * exercises
      * Q&A at Piazza
      * practice exams



Source materials are available at github repo at [https://github.com/bertdv/BMLIP](https://github.com/bertdv/BMLIP). You do not need to bother with this site. If you spot an error in the materials, please raise the issue at Piazza.  

"""

# ╔═╡ 0d016cf8-d294-11ef-0c84-336979a02dd7
md"""
## Study Guide

Slides that are not required for the exam are moved to the end of the notes and preceded by an [OPTIONAL SLIDES](#optional) header.

<p style="color:red">Please study the lecture notes before you come to class!!</p> 

Optionally, you can view the video recordings of the Q2-2023 lecture series for addional explanations. 

Then come to the class!

  * During the scheduled classroom meetings, I will not teach all materials in the lecture notes.
  * Rather, I will first discuss a summary of the lecture notes and then be available for any additional questions that you may still have.

Still got any sticky issues regarding the lecture notes?

  * Pose you question at the **Piazza site**!
  * Your questions will be answered at the Piazza site by fellow students and accorded (or corrected) by the teaching staff.

Each class also comes with a set of exercises. They are often a bit challenging and test more of your quantitative skills than you will need for the exam. When doing exercises, feel free to make use of Sam Roweis' cheat sheets for [Matrix identities](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Roweis-1999-matrix-identities.pdf) and [Gaussian identities](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/files/Roweis-1999-gaussian-identities.pdf). Also accessible from the course homepage.   

"""

# ╔═╡ 0d017b82-d294-11ef-2d11-df36557202c9
md"""
## Piazza (Q&A)

We will be using Piazza for Q&A and news. The system is highly catered to getting you help fast and efficiently from both classmates and the teaching staff. 

[Sign up for Piazza](http://piazza.com/tue.nl/winter2025/5ssd0) today if you have not done so. And install the Piazza app on your phone! 

The quicker you begin asking questions on Piazza (rather than via emails), the quicker you'll benefit from the collective knowledge of your classmates and instructors. We encourage you to ask questions when you're struggling to understand a concept—you can even do so anonymously.

We will also disseminate news and announcements via Piazza.

Unless it is a personal issue, pose your course-related questions at Piazza (in the right folder). 

Please contribute to the class by answering questions at Piazza. 

  * If so desired, you can contribute anonymously.
  * Answering technical questions at Piazza is a great way to learn. If you really want to understand a topic, you should try to explain it to others.
  * Every question has just a single students' answer that students can edit collectively (and a single instructors’ answer for instructors).

You can use LaTeX in Piazza for math (and please do so!). 

Piazza has a great ``search`` feature. Use search before putting in new questions.

"""

# ╔═╡ 0d018ee2-d294-11ef-3b3d-e34d0532a953
md"""
## Exam Guide

The course will be scored by two programming assignments and a final written exam. See the [course homepage](https://biaslab.github.io/teaching/bmlip/) for how the final score is computed.

The written exam in multiple-choice format. 

You are not allowed to use books nor bring printed or handwritten formula sheets to the exam. Difficult-to-remember formulas are supplied at the exam sheet.

No smartphones at the exam.

The tested material consists of the mandatory lecture + PP notes (+ mandatory reading assignments as assigned in the first cell/slide of each lecture notebook).

The class homepage contains two representative practice exams from previous terms. 



"""

# ╔═╡ 0d019cde-d294-11ef-0563-6b41bc2ca80f
md"""
## Preview

Check out [a recording from last year](https://youtu.be/k9DO26O6dIg?si=b8EiK12O_s76btPn) to understand what this class will be like. 

"""

# ╔═╡ 0d01a404-d294-11ef-3fe4-df9726debd05
md"""
#  $(HTML("<span id='optional'>OPTIONAL SLIDES</span>")) 

"""

# ╔═╡ 0d01b03e-d294-11ef-3b2f-53f22689075c
md"""
## Title

The slides below the `OPTIONAL SLIDES` marker are optional for the exam.  

"""

# ╔═╡ 0d01d38e-d294-11ef-3d7a-791fdf340771
open("../../styles/aipstyle.html") do f
    display("text/html", read(f,String))
end

# ╔═╡ Cell order:
# ╟─0cfd4bc0-d294-11ef-3537-630954a9dd27
# ╟─0cffef7e-d294-11ef-3dd5-1fd862260b70
# ╟─0d013750-d294-11ef-333c-d9eb7578fab2
# ╟─0d0142b6-d294-11ef-0297-e5bb923ad942
# ╟─0d015ab4-d294-11ef-2e53-5339062c435c
# ╟─0d016cf8-d294-11ef-0c84-336979a02dd7
# ╟─0d017b82-d294-11ef-2d11-df36557202c9
# ╟─0d018ee2-d294-11ef-3b3d-e34d0532a953
# ╟─0d019cde-d294-11ef-0563-6b41bc2ca80f
# ╟─0d01a404-d294-11ef-3fe4-df9726debd05
# ╟─0d01b03e-d294-11ef-3b2f-53f22689075c
# ╠═0d01d38e-d294-11ef-3d7a-791fdf340771
