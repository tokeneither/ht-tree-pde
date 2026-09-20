# Focused prior-art audit: certificates, relaxation gaps, and parameter reuse

Date: 20 September 2026. Scope: a focused primary-source comparison with the manuscript and its supporting derivations. This is an assessment of mathematical overlap, not a priority certificate or an exhaustive systematic review. No submission or external communication was performed.

## Assessment

The most defensible mathematical contribution remains the **quantitative separation after optimizing both the scalar center and a spectral projection, while charging exact omitted energy**. The balanced metric is an elementary, useful construction; the underlying convexification and the all-edge norm duality already exist. The parameter-reuse proposal is mathematically valid, but its basic perturbation inequality is too general and elementary to carry a novelty claim. It becomes a computational contribution only if its certified coverage and total cost survive strong offline–online and warm-reuse baselines.

| Current claim | Closest inspected result | Assessment and manuscript implication |
|---|---|---|
| Rank-constrained quadratic supremum equals a sum of the largest squared singular values | Grussler–Giselsson, Theorem 1 and equations (14)–(15); Andersson–Carlsson–Olsson, Theorem 1 | Known. Cite as a building block. |
| Optimize the center of a product-metric bound | Low-rank inducing Frobenius norm convexification after rank-preserving whitening; Carlsson, Theorem 5.1 | Equivalent to a known convex-envelope relaxation under a change of coordinates. The PDE metric and its use in tree exclusion may be contributions; the convexification is not new. |
| Larger valid product minorants improve the bound at every center | Displacement supremum with a larger quadratic penalty | Elementary monotonicity. Useful lemma, weak standalone novelty. |
| Balanced shifted geometric metric dominates scalar coercivity | Commuting AM–GM inequality applied after balancing spectral minima | A transparent tailored construction. No exact duplicate located, but this is not evidence of deep or established novelty. |
| Sharp scalar-plus-exact-tail gap asymptotic to `3 / 2^(2/3) * k^(2/3)` | Known low-rank convex envelopes explain the scalar objective; inspected papers do not provide this projection-plus-tail comparison | Strongest new-result candidate. The comparator must remain explicit and restricted. |
| Ordinary Dirichlet Laplacian has an unbounded compression-resistant gap | Weighted low-rank approximation and Sobolev tensor approximation are established; exact proposed construction not located | Potentially useful PDE corollary. It uses a resolution-dependent forcing; do not imply a fixed smooth PDE forcing has this behavior. |
| Joint allocation over every edge is a norm relaxation | Tomioka–Suzuki, Lemma 1, latent/overlapped Schatten duality | Known norm identity, including the relevant `p=1, q=infinity` case. |
| Fully optimized all-edge allocation has gap `n` on matrix multiplication | Derksen, Theorem 1.9 and Corollary 1.10 give the tensor and its injective norm; latent norm duality gives the relaxation interpretation | The precise allocation statement is a candidate corollary, but readily follows from established structures. Do not present the tensor or its norm as new. |
| Degree-six SOS is exact on that example | Barak–Moitra and Foster–Risteski already study degree-six tensor norm relaxations | The explicit Lagrange-identity certificate is a useful explanatory separation. It does not establish a new general SOS hierarchy or an efficient HT algorithm. |
| A gap certifies a region of unchanged best tree | Support-function Lipschitz continuity; safe-screening methodology | The HT application can be useful; the elementary continuity lemma and “margin exceeds perturbation” principle are not a substantial independent theorem. |
| Reuse a tree between related PDE instances | HiST version 2, especially Figures 12 and 14; reduced-basis offline–online methods | Empirical reuse already exists. The candidate distinction is global certification of the family comparison. |
| `U-L` bounds excess approximation error | Ritz identity and any valid upper/lower objective bounds | Standard certificate consequence. Include for usability, not as a headline novelty claim. |

## 1. Exact relationship to established low-rank convexification

[Grussler and Giselsson, *Low-Rank Inducing Norms with Optimality Interpretations*, 2018](https://arxiv.org/pdf/1612.03186) define truncated dual norms in equations (7)–(9). Their Theorem 1, equations (14)–(15), identifies the conjugate and biconjugate of a unitarily invariant norm composed with an increasing convex scalar function, with a rank constraint. Choosing the squared Frobenius norm gives precisely the envelope needed here. Their discussion of splitting a convex objective before convexifying is also directly relevant.

The following mapping is our derivation for this project. Put

\[
 q_C(X)=\langle X,CX\rangle,\qquad
 h_B(X)=\langle X,BX\rangle+\iota_{\operatorname{rank}(X)\le r},
 \quad C=A-B\succeq0.
\]

For `B(X)=LXR`, the invertible map `W(X)=L^(1/2) X R^(1/2)` preserves rank. Consequently

\[
 h_B^{**}(X)=\|W(X)\|_{F,r*}^{2},\qquad
 h_B^*(2F)=\sum_{j\le r}\sigma_j(W^{-1}F)^2.
\]

Here `||.||_(F,r*)` is the known low-rank inducing Frobenius norm. The centered objective is

\[
 \Phi_B(Y)=q_C(Y)+h_B^*(2F-2CY).
\]

With `p=2CY`, `q_C^*(p)=p^T C^dagger p/4` on `range(C)`, and infinity otherwise. Infimal convolution therefore gives

\[
 \inf_Y\Phi_B(Y)
 =(q_C+h_B^{**})^*(2F)
 =\sup_X\{2\langle F,X\rangle-q_C(X)-h_B^{**}(X)\}.
\]

The final expression is the existing rank-convexification framework applied to the `B` portion of the energy. It also handles singular `C`; no unjustified ordinary inverse is required. This is an equivalence of the fully optimized value. Evaluating only a few centers remains a valid, potentially much cheaper certificate and need not solve this convex problem.

[Andersson, Carlsson, and Olsson, *Convex envelopes for fixed rank approximation*, 2017](https://link.springer.com/article/10.1007/s11590-017-1146-5), Theorem 1, gives an explicit singular-value formula for the envelope of rank-constrained squared approximation error. This is an additional close reference, especially for any exact center-optimization formula. [Carlsson, *On Convex Envelopes and Regularization of Non-convex Functionals Without Moving Global Minima*, 2019](https://arxiv.org/pdf/1811.03439), Theorems 3.3 and 5.1, describes retaining a positive quadratic remainder while convexifying a quadratic-plus-nonconvex term. The PDF text extraction corrupts some Loewner-order symbols; the proof of Theorem 5.1 explicitly uses positivity of the quadratic remainder.

**Consequences.** Replace any suggestion of a newly discovered convexification with a statement that operator-informed, rank-preserving metrics give a useful certificate inside established convex-envelope machinery. A scalar certificate applied **after product whitening** is the product certificate in other coordinates. Our separation cannot honestly claim superiority over all scalar methods allowing arbitrary such preprocessing.

## 2. Balanced metric and compression-resistant separation

The balanced metric needs only the scalar inequality `2 sqrt(ab) <= a+b`, because the left and right tensor-factor operators commute. The shift makes the two smallest factor eigenvalues equal, so the geometric mean is at least the optimal original scalar coercivity. This observation does not require a new theorem on noncommuting matrix means. Focused searches for Sylvester/Kronecker geometric preconditioners did not locate the identical shift-and-certificate statement, but absence in these searches establishes no priority.

The compression-resistant example is appreciably more specific. Existing envelope formulas do not, by themselves, optimize a product spectral subspace while charging exact omitted solution energy. The project proves an exact suffix reduction followed by a sharp asymptotic balance. That is the clearest candidate for a new quantitative theorem among the current claims.

State the admissible comparator in the theorem itself:

1. one retained product spectral space, allowing arbitrary row and column index subsets;
2. the largest scalar minorant on that space;
3. arbitrary center optimization inside that space;
4. the exact omitted energy added as a tail allowance.

It does **not** encompass multiblock scalar bounds, arbitrary rank-preserving preconditioning, optimized general positive metrics, or nonlinear approximation algorithms. The Dirichlet forcing changes with resolution and gives one unit of energy to each diagonal spectral component. This is a worst-case certificate-separation family, not convergence of a single fixed forcing under mesh refinement. These qualifications improve the theorem's credibility without weakening its precise statement.

No identical `k^(2/3)` result was found in the inspected primary papers or focused phrase searches. This is a **new-result candidate**, not a verified priority claim. A specialist should check its closest convex-envelope and weighted-approximation antecedents before submission.

## 3. All-edge allocation: known norm, explicit limitation

[Tomioka and Suzuki, *Convex Tensor Decomposition via Structured Schatten Norm Regularization*, NeurIPS 2013](https://proceedings.neurips.cc/paper_files/paper/2013/file/109a0ca3bc27f3e96597370d5c8cf03d-Paper.pdf), Section 2 and Lemma 1, equations (2) and (4), already identify duality between overlapped Schatten `p/q` norms and latent Schatten `p*/q*` norms. Taking `p=1, q=infinity` yields the duality between the maximum unfolding nuclear norm and an infimum of sums of unfolding spectral norms. This is exactly the norm arising after optimizing the scalar allocation weights here.

In project notation the elementary weight optimization gives

\[
 U_{\rm all}(F)=
 \left(\inf_{\sum_e F_e=F}\sum_e\|F_e^{(e)}\|_{\rm op}\right)^2
 =\left(\sup_{\max_e\|Z^{(e)}\|_*\le1}\langle F,Z\rangle\right)^2.
\]

[Derksen, *On the Nuclear Norm and the Singular Value Decomposition of Tensors*](https://arxiv.org/pdf/1308.3860), Section 1.5, Theorem 1.9 and Corollary 1.10, gives the matrix-multiplication tensor and its injective norm one (and tensor nuclear norm `pqr`). The proposed equal-sided example combines this old tensor with a simple unfolding-norm calculation. The full-allocation witness is still worth stating because it shows that optimizing all forcing allocations does not repair the compatibility gap. However, a referee may view it as a short consequence of those two established ingredients rather than a main theorem of sufficient depth by itself.

The degree-six identity in the [paper draft](../paper/draft.pdf) is an explicit polynomial certificate for the **squared trilinear objective**. [Foster and Risteski, *Sum-of-squares meets square loss*](https://arxiv.org/pdf/1905.13283), Section 2.1, equations (6)–(8), already defines SoS injective/nuclear norm relaxations and uses degree six. Their objective is expressed through a linear tensor functional of pseudo-moments; this should not be silently identified with every formulation of the squared-objective hierarchy. [Barak and Moitra, *Noisy Tensor Completion via the Sum-of-Squares Hierarchy*](https://arxiv.org/pdf/1501.06521) is a further direct antecedent. Cite this work and describe our identity as a concrete witness of the limitation of edgewise convexification. Generic HT factor-SOS efficiency remains unproved.

## 4. Parameter-region certification and the closest methods

For fixed `A` and a family closed under real scaling, the project's own identity is

\[
 s_T(f):=\sqrt{G_T(f)}=
 \sup_{x\in K_T\setminus\{0\}}\frac{|\langle f,x\rangle|}{\|x\|_A}.
\]

This is a support-function/norm construction. Its one-Lipschitz property in `||.||_(A^-1)` follows immediately. Thus a certified margin gives a robust classification of the best family, with an ellipsoid after substituting affine forcing and a computable inverse-metric upper bound. The factor-of-two symmetric perturbation margin, and its improvement using the stored feasible direction, should be presented as practical consequences rather than a deep new continuity result.

[Ndiaye, Fercoq, Gramfort, and Salmon, *Gap Safe Screening Rules for Sparsity Enforcing Penalties*, JMLR 2017](https://www.jmlr.org/papers/volume18/16-577/16-577.pdf), Proposition 5 and Theorem 6, provides a safe region and an exclusion test. Equation (11) derives a dual-solution ball from a duality gap; Section 3.3 combines screening and active warm starts along a parameter path. This is a methodological ancestor, **not an identical theorem**: our region is in forcing/parameter space and preserves the ordering of nonconvex rank-family optima. It does not locate a convex dual optimizer via strong concavity.

[HiST, inspected version 2 of 28 July 2026](https://arxiv.org/html/2603.27856v2), explicitly studies transferring a discovered tensor-network structure to related data. Figures 12 and 14 compare fixed structures across time steps and show both useful transfer and failures during changing transients. That is direct prior art for the empirical reuse premise. Our possible distinction is a global certificate over the prescribed families, not the idea of reusing a structure.

[Rozza et al., *Basic Ideas and Tools for Projection-Based Model Reduction of Parametric Partial Differential Equations*](https://arxiv.org/pdf/1911.08954), Sections 1.3.4 and 1.3.6, equations (26)–(31), describes affine operator/forcing expansions and moving large-dimensional work offline. Section 1.4 treats residual error estimation. These are the natural competing computational principles in the proposed many-query setting; related reduced-basis work predates this chapter by decades.

## 5. Required strong comparisons for the reuse experiment

The following are experimental design deductions, not claims attributed to a particular paper.

**Exact affine-forcing baseline.** For fixed `A` and

\[
 f(\mu)=\sum_{q=0}^{p}\theta_q(\mu)f_q,
 \quad Au_q=f_q,
 \qquad u(\mu)=\sum_{q=0}^{p}\theta_q(\mu)u_q.
\]

Only `p+1` basis solves are needed. Once their cost and storage are paid, the online PDE solve disappears exactly. A shared factorization or spectral diagonalization is also reused. In compressed arithmetic, linear combinations may grow ranks and require rounding; include this cost for both methods. Comparing certificate reuse with a fresh PDE solve on every query would miss this baseline and exaggerate the avoided work.

**Tree/factor reuse baseline.** Cache a previously chosen tree and its tensor factors, warm-start the fixed-tree fit, and periodically or adaptively refresh the structure search. Evaluate the same final approximation-quality metric. A stronger data-side baseline reconstructs the affine solution and performs cached cut computations only when necessary. Measure wrong selections/quality losses as well as time; heuristics and globally certified outputs have different guarantees.

**The same reuse mechanism on every valid seed certificate.** The region lemma applies equally to a product certificate, a scalar certificate, a solve-first certificate, or any other valid lower/upper bounds. Attach the same region rule to these competitors and account for their setup costs. Otherwise a speedup demonstrates generic amortization rather than a product-metric advantage.

**A certificate-assisted warm heuristic.** First try the previous tree and its factors; compute a cheap lower gain; update competitor bounds only if the existing certificate no longer proves the requested guarantee. This hybrid may be the strongest practical algorithm, even if it changes the original “select before solving” narrative.

Record total wall time, offline time, online time, memory, certified coverage, fallback/abstention rate, and excess error at fixed tolerances. Include domains crossing tree transitions, genuinely unseen parameter points, and the same parameter sequence for every method. A small ball selected because it was already certified gives a demonstration of correctness, not evidence of useful real-world coverage. State whether the measured output is just a tree label, an approximation, or a complete PDE solution.

## 6. Implications for the next manuscript

For a **SISC-oriented paper**, the persuasive contribution would be a complete certified selection/reuse algorithm that has useful coverage or accuracy guarantees at reasonable total cost against the preceding baselines. It need not beat an unverified heuristic in every timing, but the extra guarantee must solve a credible computational problem. Artificially repeating cold solves or exhaustive searches is insufficient.

For a **SIMAX-oriented paper**, the center should be exact relationships among relaxation classes and sharp separation results, especially the projection-plus-exact-tail theorem. Known convexification, norm duality, and the matrix-multiplication example must be openly identified. More general spectral-growth laws or a sharp characterization of which product metrics close the gap would strengthen that route, provided the additional mathematics is substantive and independently checked.

Recommended claim wording: “We specialize established low-rank convexification to operator-informed product metrics, prove quantitative separations from a specified scalar-plus-tail class, and study certified reuse for HT-family selection.” Avoid “a fundamentally new rank relaxation,” “first tree reuse method,” “all scalar methods,” and “all edge combinations fail” without the precise scope of the operator allocation.

The audit found concrete overlaps that require careful attribution. It did not find an exact antecedent for the sharp compression-resistant separation, and it does not establish that no such antecedent exists.
