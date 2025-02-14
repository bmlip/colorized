


















# Slider server (for live `@bind` interaction)
This whole part is optional, the website will work without it. 

We are running a PlutoSliderServer (PSS) to support live instant interactions with `@bind` on the website. This is also used by computationalthinking.mit.edu and featured.plutojl.org. Read the PSS README.md to learn more!

## SURF server
The server is hosted on **SURF Research Cloud**, they provided Fons with credits an account. Our contact is Han Verbiesen (ask Fons for his email, or contact hpcsupport@tue.nl). On the SURF dashboard: you need these things (create in this order):
- Account access (need to request this), Collaborative Organisation (we have one called "BIASLab BMLIP course development (Pluto.jl, Julia)") and a Wallet with credits (we started with 5000, valid until 31-03-2025).
- A reserved IP address. Currently 145.38.187.167
- A Workspace: I used "Ubuntu 22 SUDO", 2 core 16GB RAM, linked to the reserved IP.

## Domain
To get https we need a web domain. Not sure yet!

## Setup
**The current setup is exactly the "Sample setup" from the PSS readme.**

The sample PSS setup works perfectly. The port 8080 is not available, so we have an nginx proxy from port 8080 -> 80 (also from the PSS readme sample instructions).














