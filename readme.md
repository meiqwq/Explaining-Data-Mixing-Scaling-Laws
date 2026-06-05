# Explaining Data Mixing Scaling Law

Fitting data-mixture scaling laws that predict per-domain training loss as a function of mixture weights, model size, and token count. Given a learned law, we can compute the *optimal mixture* $h^*$ that minimizes average loss across domains.

---

## Project Structure

```
.
├── data/
│   ├── dmsl_llm_slimpajama.csv   # SlimPajama LLM data (7 domains)
│   ├── test_pile_loss_1B.csv      # Pile loss data (4 domains)
│   ├── test_mixture_1B.csv       # Pile mixture weights
│   └── eval.jsonl                # Evaluation data
├── fitting_7domains.py           # Main: 7-domain law (JAX/GPU, Eq. model)
├── fitting_4domains.py           # 4-domain law (numpy/scipy, law2)
├── fitting_17domains_eq3.py      # 17-domain law (numpy/scipy, law2init)
├── fitting_17domains_eq4.py      # 17-domain law (numpy/scipy, law1)
├── optimal_mixture.py             # CLI demo: find $h^*$ from fitted params
└── utils/
    ├── datas/
    │   ├── domain4.py             # 4-domain Pile data loader
    │   └── domain17.py           # 17-domain Pile data loader
    └── fitting_algos/
        ├── law1.py               # Basic implicit-layer model
        ├── law2.py               # Per-sample N/M model (vectorized)
        ├── law2init.py           # Initialized variant of law2
        └── law_exp.py            # Exponential variant
```

---

## The Scaling Law Model

For a $K$-domain mixture with weight vector $\mathbf{h}$ ($\sum_i h_i = 1$), model size $N$ (M params), and $D$ (B tokens), the per-domain loss is modeled as:

$$L_i(h, N, D) = c_i \cdot {x_i}^*(h)^{-b_i} + A_i \cdot (D \cdot h_i)^{-a_i} + E_i$$

| Term | Meaning |
|------|---------|
| $c_i \cdot x^{-b_i}$ | **Capacity term** — diminishing returns from compute allocation $x_i^*$ |
| $A_i \cdot (D \cdot h_i)^{-a_i}$ | **Noise term** — statistical inefficiency when a domain is under-represented |
| $E_i$ | **Irreducible error** — lower bound on loss for domain $i$ |

**$x^*(h)$** is the KKT solution of the inner problem:

$$\min_{x} \sum_i h_i \cdot c_i \cdot x_i^{-b_i} \quad \text{s.t.} \quad \sum_i x_i = N + (K-1) \cdot H, \quad x_i \ge H$$

— meaning the model allocates compute across domains to minimize the capacity-weighted loss, subject to a global capacity slack $H$.

The fitted parameters per domain are: $b, c, A, a, E$ ($5 \cdot K + 1$ scalars total).

---

## Workflows

### 7-Domain LLM (SlimPajama, JAX/GPU) — `fitting_7domains.py`

Uses the full model above with JAX-accelerated bisection and exact gradients.

```bash
# GPU recommended; falls back to CPU if unavailable
python fitting_7domains.py
```

Output includes:
- Train/Val MRE, MAE, RMSE
- Fitted per-domain parameters $(b_i, c_i, A_i, a_i, E_i)$ and global $H$
- Optimal mixture $h^*$ at the current $(N, D)$ scale
- `DOMAIN_PROPORTIONS` dict (RedPajama naming, sums to 1)

### 4-Domain Pile (numpy/scipy) — `fitting_4domains.py`

Fits the law with per-sample $N[i]$ and $M[i]$ using `law2`. Each sample has a different model size and token count (supports 90M–440M models, 3B–40B tokens).

```bash
python fitting_4domains.py
```

Domains: `Github_text_document`, `Gutenberg_text_document`, `StackExchange_text_document`, `Wikipedia_text_document`

### 17-Domain Pile (numpy/scipy) — `fitting_17domains_eq1.py` / `fitting_17domains_eq3.py`

Same Pile data with 17 domains, using `law2init` (Eq. 3) or `law1` (Eq. 1) respectively.

```bash
python fitting_17domains_eq3.py   # law2init
python fitting_17domains_eq1.py   # law1
```

---

## Optimal Mixture — `optimal_mixture.py`

Given fitted parameters from any workflow, find the optimal mixture $h^*$ that minimizes mean loss across all $K$ domains:

$$\min_{h} \frac{1}{K} \sum_i L_i(h, N, D) \quad \text{s.t.} \quad \sum_i h_i = 1, \quad h_i \ge 10^{-4}$$

Uses **Basin-Hopping + L-BFGS-B** with softmax reparameterization (unconstrained logits $z \in \mathbb{R}^K$).

```bash
python optimal_mixture.py
```

Prints $h^*$ for each domain at $N=1.3$B params, $D=30$B tokens.

To use with your own fitted parameters, replace the arrays in the `__main__` block:

```python
b_fit = np.array([...])   # shape (K,)
c_fit = np.array([...])
A_fit = np.array([...])
a_fit = np.array([...])
E_fit = np.array([...])
H_fit = ...  # scalar
N     = 1300  # model size in millions
D     = 30    # training tokens in billions

h_star, loss_star = find_optimal_mixture(b_fit, c_fit, A_fit, a_fit, E_fit, H_fit, N, D)
```

---

## Dependencies

```
numpy
pandas
scipy
jax / jaxlib          # GPU-accelerated fitting (fitting_7domains.py)
matplotlib            # plotting (optional)
cvxpy                 # convex optimization helpers (law1)
```

Install:
```bash
pip install numpy pandas scipy jax matplotlib cvxpy
```