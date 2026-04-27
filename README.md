# 2026-MLP-causal-inference-workshop

# From Prediction to Impact: Machine Learning Meets Econometrics for Causal Inference in Organizations

A hands-on course on estimating causal treatment effects from observational HR data in quasi-experimental setting.

## Before the Workshop

Hi! Thanks a lot for signing up, we are thrilled to meet you in Prague soon. To let us know a bit more about your background, would you mind taking 5 minutes to complete 
<a href="https://forms.cloud.microsoft/e/uasEWsQhtS" target="_blank" rel="noopener noreferrer">
  📄 this form?</a> It will help us optimize the workshop's pace. It would be great if you can submit your response by 1 May 2026. Thank you!

## What You'll Learn

This course walks through a real-world people analytics use case: **evaluating whether a manager training program causally improves leadership outcomes and team retention.** You'll learn to:

- **Frame the causal question** — discuss possible identification strategies in similar settings, choose the right estimand (ATE vs. ATT)  given the context and research question and understand when and why they diverge
- **Diagnose before modeling** — multicollinearity (VIF/GVIF), covariate overlap, positivity, and estimand feasibility
- **Estimate with IPTW** — build, stabilize, trim, and diagnose inverse probability of treatment weights, then verify covariate balance pre- vs. post-weighting
- **Model survey outcomes** — IPTW + covariate-adjusted GEE with cluster-robust standard errors
- **Model retention (time-to-event)** — IPTW-weighted **Kaplan–Meier** and **RMST** (non-parametric) and **Cox PH with time interactions** (inferential, when proportional hazards is violated)
- **Check robustness** — cluster-robust **DoubleML (PLR)** as an alternative identification strategy, and **E-value sensitivity analysis** for unmeasured confounding
- **Explore heterogeneity** — use a **Causal Forest (DML)** to surface conditional average treatment effects (CATE) along actionable moderators — who benefits most
- **Communicate results** — translate technical findings into stakeholder-ready recommendations with transparent caveats
- **Bonus (to explore after the workshop)** - develop a set of skills for an AI Agent to conduct similar analyses
---

## Prerequisites

- Familiarity with regression concepts (OLS, logistic regression)
- Basic understanding of causal inference concepts (confounding, selection bias) is helpful but not required — the course materials cover these
- Working knowledge of Python (pandas, numpy) or similar language (R) also helpful but not required

---

## Get Started

**Option 1:** Google Colab

No local installation required. Everything runs in your browser.

<a href="https://colab.research.google.com/github/mlpost/2026-siop-causal-inference-master-tutorial/blob/main/causal_inference_workshop.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Option 2:** HTML Report (no coding environment)

See teaching content and output only (no code blocks)

<a href="https://mlpost.github.io/2026-causal-inference-tutorial-siop-ml-prague/causal_inference_workshop_no_code.html" target="_blank" rel="noopener noreferrer">
  📄 View HTML Report
</a>
