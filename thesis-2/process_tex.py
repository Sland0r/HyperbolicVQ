import re
import sys

filename = '/home/acolombo/VAEs/thesis-2/title-page-ai.tex'

with open(filename, 'r') as f:
    content = f.read()

# 1. Extract and remove sections to move
forward_failure_pattern = re.compile(r'(\\section\{Forward failure: non-telescoping residual cascade\}.*?)(?=\\subsection\{Hyperbolic Residual Aggregation \(HRA\)\})', re.DOTALL)
backward_failure_pattern = re.compile(r'(\\section\{Backward failures: wrong and cumulative gradient\}.*?)(?=\\subsection\{The Discounted Hyperbolic Straight-Through Estimator \(d-HSTE\)\})', re.DOTALL)

forward_failure_match = forward_failure_pattern.search(content)
backward_failure_match = backward_failure_pattern.search(content)

forward_failure_text = forward_failure_match.group(1) if forward_failure_match else ""
backward_failure_text = backward_failure_match.group(1) if backward_failure_match else ""

content = forward_failure_pattern.sub('', content)
content = backward_failure_pattern.sub('', content)

# 2. Change subsections to sections
content = content.replace(r'\subsection{Hyperbolic Residual Aggregation (HRA)}', r'\section{Hyperbolic Residual Aggregation}')
content = content.replace(r'\subsection{The Discounted Hyperbolic Straight-Through Estimator (d-HSTE)}', r'\section{The Discounted Hyperbolic Straight-Through Estimator}')

# 3. Insert into appendix
appendix_text = "\n\\chapter{Method Failures Analysis}\n" + forward_failure_text + "\n" + backward_failure_text + "\n"
content = content.replace(r'\chapter{Optional appendix}', r'\chapter{Optional appendix}' + appendix_text)

# 4. Modify Method introduction
method_intro_old = r"""This chapter develops the residual-quantization module at the centre of this thesis: \emph{Geometry-aware Hyperbolic Residual Quantization} (GHRQ). Building on the background of Chapter~\ref{chap:background}, we lift RVQ onto the Poincar\'e ball and ask what it takes to make residual quantization a genuine operation \emph{on} the manifold rather than a Euclidean procedure performed on hyperbolic coordinates. Prior hyperbolic residual quantizers \cite{hrqvae,hyprqvae_recsys} place the codes inside the ball and assign them by geodesic distance, but keep the Euclidean training machinery, copying gradients through the Identity Straight-Through Estimator (I-STE) and transliterating the residual recursion into its M\"obius symbols. We will address this as the \emph{naive hyperbolic} approach. We first show (\S\ref{sec:hrq} and~\S\ref{sec:firewall}) that this naive lift leaves the two main operations of residual quantization (both in forward and backward passes) geometrically wrong at once: \emph{(i)} the residual recursion itself, which on the ball no longer telescopes, drifts away from the true reconstruction error; and \emph{(ii)} the estimated gradient from the quantizer is the wrong vector and accumulates across stages, exploding with depth. These two failures become decisive as the depth grows, motivating the two components of GHRQ which we then construct in turn: the \emph{Hyperbolic Residual Aggregation} (HRA) method that repairs the cascade (\S\ref{sec:residagg}) and a block-level \emph{discounted hyperbolic straight-through estimator} (d-HSTE) that repairs the gradient (\S\ref{sec:hste}). We close by assembling the full module (\S\ref{sec:ghrq})."""

method_intro_new = r"""This chapter develops the Geometry-aware Hyperbolic Residual Quantization module. The pipeline consists of two main components that correctly lift residual quantization onto the Poincar\'e ball: Hyperbolic Residual Aggregation, which provides a geometrically correct coarse-to-fine decomposition in the forward pass, and the discounted Hyperbolic Straight-Through Estimator, which routes the correct gradient back to the encoder during the backward pass."""

if method_intro_old in content:
    content = content.replace(method_intro_old, method_intro_new)
else:
    print("Warning: old method intro not found exactly.")

# 5. Insert motivation into Introduction
motivation_text = "\n\nPrior hyperbolic residual quantizers place the codes inside the ball and assign them by geodesic distance, but keep the Euclidean training machinery. This naive lift leaves the two main operations of residual quantization geometrically wrong at once: first, the residual recursion itself, which on the ball no longer telescopes, drifts away from the true reconstruction error; second, the estimated gradient from the quantizer is the wrong vector and accumulates across stages, exploding with depth. These two failures become decisive as the depth grows, motivating the two components of our Geometry-aware Hyperbolic Residual Quantization module: the Hyperbolic Residual Aggregation method that repairs the cascade, and a block-level discounted hyperbolic straight-through estimator that repairs the gradient."

intro_target = r"leveraged the curvature of the space during codebook updates."
content = content.replace(intro_target, intro_target + motivation_text)

# 6. Global abbreviation replacement
abbrev_map = {
    r'\bd-HSTE\b': 'discounted Hyperbolic Straight-Through Estimator',
    r'\bI-STE\b': 'Identity Straight-Through Estimator',
    r'\bHRA\b': 'Hyperbolic Residual Aggregation',
    r'\bGHRQ\b': 'Geometry-aware Hyperbolic Residual Quantization',
    r'\bRVQ-VAEs?\b': 'Residual Quantized Variational Autoencoders',
    r'\bRQ-VAEs?\b': 'Residual Quantized Variational Autoencoders',
    r'\bVQ-VAEs?\b': 'Vector-Quantized Variational Autoencoders',
    r'\bRVQ\b': 'Residual Vector Quantization',
    r'\bSTE\b': 'Straight-Through Estimator',
    r'\bEMA\b': 'Exponential Moving Average'
}

for abbr, full_name in abbrev_map.items():
    content = re.sub(abbr, full_name, content)

with open(filename, 'w') as f:
    f.write(content)

print("Processing complete.")
