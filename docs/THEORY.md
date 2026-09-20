# Mathematical guide

Full proofs and references are in the [paper draft](../paper/draft.pdf).
This guide identifies the statements tested by the three retained experiments.

## 1. Compare approximation families without the full solution

Let `A` be a finite-dimensional symmetric positive definite operator, `Au=f`,
and let `K_T` be a closed tensor family with prescribed ranks, closed under real
scaling. Define

\[
 g_f(x)=2\langle f,x\rangle-\langle x,Ax\rangle,
 \qquad G_T(f)=\max_{x\in K_T}g_f(x).
\]

The energy identity gives

\[
 \min_{x\in K_T}\|u-x\|_A^2=\|u\|_A^2-G_T(f).
\]

The unknown common solution energy cancels when comparing families.
A feasible tensor gives `L_T <= G_T`; a valid relaxation gives `G_T <= U_T`.
If `L_* > max_{T != *} U_T`, the best family is certified. A local optimizer
can supply the feasible tensor; it need not solve the nonconvex problem globally.
If intervals overlap, the method may abstain.

For one unfolding, a positive rank-preserving product metric `B <= A` gives,
with `C=A-B`, the bound

\[
 G_T(f)\le \langle y,Cy\rangle+
 \sum_{j=1}^{r}\sigma_j\!\left(B^{-1/2}(f-Cy)\right)^2
\]

for every center `y`. Here `B^{-1/2}` acts by left/right whitening of the
unfolding. Combining bounds from constrained cuts yields bounds for an HT
family. The convexification behind this formula is established mathematics;
see the [prior-art audit](RELATED_WORK.md).

The [8D study](../results/ht/README.md) tests 315 balanced trees with rank at most
two on every nonroot edge, including leaves. Fourteen non-tie cases certify
with one fit each; the equal-gain control abstains. The cached solve-and-round
heuristic is faster in these small dense problems.

## 2. Compression-resistant separation

For a rank-one matrix family, take

\[
 A_L=A_R=\tfrac12\operatorname{diag}(1^2,\ldots,k^2),
 \qquad A(X)=A_LX+XA_R,\qquad F=\operatorname{diag}(1,\ldots,k).
\]

The exact rank-one gain and a product-metric upper bound are both one.
In contrast, the best scalar relaxation in the following class satisfies

\[
 U_{\mathrm{scalar+tail}}^\star
 =\left(\frac{3}{2^{2/3}}+o(1)\right) k^{2/3}:
\]

- choose one arbitrary retained product spectral subspace;
- use its best scalar coercivity and optimize the center;
- add the exact omitted solution energy.

Each omitted diagonal mode costs one unit of energy. The proof reduces the
optimal retained set to a suffix, then balances its scalar bound against the
tail. The comparator excludes multiblock bounds and arbitrary product whitening.
It is a separation of relaxation quality, not a runtime theorem.

The [gap data](../results/gap/results.json) contain primal/dual and exhaustive
subset checks. The [Dirichlet audit](../results/gap/dirichlet.json) checks the
analog for ordinary finite-difference eigenvalues. Its forcing depends on grid
resolution; this is not a mesh-refinement assertion for one fixed smooth forcing.

## 3. Fixed-operator parameter reuse

For a family closed under scaling,

\[
 \sqrt{G_T(f)}=
 \sup_{x\in K_T\setminus\{0\}}
 \frac{|\langle f,x\rangle|}{\|x\|_A},
 \qquad
 |\sqrt{G_T(f+h)}-\sqrt{G_T(f)}|\le\|h\|_{A^{-1}}.
\]

Thus a margin `sqrt(L_*) - max sqrt(U_T) > 2 delta` preserves the winning
family whenever `||h||_(A^-1) <= delta`. A computable inverse-metric bound for
affine forcing gives a region in parameter space. Homogeneous scaling and
stored feasible directions can improve the practical transport bounds.
This continuity principle is elementary; its computational use requires strong
comparisons, including the same reuse policy applied to solve-first certificates.

The [reuse study](../results/reuse/README.md) retains all 480 queries, nine
sequences, six methods and three timing repetitions. Direct reuse certifies all
queries, but the offline basis-solve method with the same reuse policy is faster
on every tested sequence. Only the forcing varies; general variable-operator
reuse is outside the demonstrated scope.

## 4. A usable guarantee when a unique family cannot be certified

For any actual feasible output `x_hat`, set `L=g_f(x_hat)` and `U=max_T U_T`.
Then

\[
 \|u-\hat x\|_A^2-
 \min_{T,\;x\in K_T}\|u-x\|_A^2\le U-L.
\]

This controls excess squared energy error even if several families remain
competitive. It follows directly from the energy identity. The implementation
uses ordinary floating-point checks; machine-verifiable interval certificates,
adaptive rank allocation, and unrestricted tree topologies are not claimed.
