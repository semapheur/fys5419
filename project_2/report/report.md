---
title: Quantum Phase Estimation of Energy Eigenvalues
subtitle: FYS5419 - Project 2
authors:
  - name: Insert Name
site:
  template: article-theme
exports:
  - format: pdf
    template: ../../report_template
    output: report.pdf
    showtoc: true
math:
  # Note the 'single quotes'
  '\argmin': '\operatorname{argmin}'
  '\R': '\mathbb{R}'
  '\N': '\mathbb{N}'
  '\unitvec': '{\hat{\mathbf{#1}}}'
bibliography: references.bib
abstract: |
  We apply the quantum phase estimation (QPE) algorithm to estimate eigenvalues of one- and two-qubit Hamiltonians. QPE is implemented to extract both ground and excited state energies, and its performance is compared against the variational quantum eigensolver. Results highlight the strengths of QPE in resolving full spectral information and achieving high-precision estimates with minimal repetition, while also illustrating practical limitations related to circuit depth and state preparation.
---

# Introduction

This project explores the application of the quantum Fourier transform (QFT) in the quantum phase estimation (QPE) algorithm. Specifically, QPE is employed to estimate the eigenvalues of one- and two-qubit Hamiltonians. The accuracy and performance of QPE are benchmarked against results obtained using the variational quantum eigensolver (VQE).

The report begins with a theoretical overview of the QFT and its role within the QPE framework. This is followed by a presentation of the numerical results obtained from implementing QPE on small-scale Hamiltonians. Finally, the relative advantages and limitations of QPE and VQE are discussed, with attention to algorithmic precision and circuit depth.

All numerical experiments were performed using Python, and the source code is available at: [https://github.com/semapheur/fys5419](https://github.com/semapheur/fys5419).

# Theory and Method

## Quantum Fourier Transform

Let ${}^\P \mathcal{H}^{\otimes n}$ denote the $n$-qubit Hilbert space, with computational basis $\set{\ket{j}}_{j=0}^{N-1}$ for $N = 2^n$. The quantum Fourier transform (QFT) is a linear tranformation defined on computational basis states $\ket{j}\in {}^\P \mathcal{H}^{\otimes n}$ by

$$
\begin{equation*}
  \ket{j} \mapsto \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{i2\pi jk/N} \ket{k}
\end{equation*}
$$

By linearity, the QFT maps an arbitrary qubit state $\ket{\psi} = \sum_{j=0}^{N-1} \psi_j \ket{j} \in {}^\P \mathcal{H}^{\otimes n}$ as

$$
\begin{equation*}
  \ket{\psi} \mapsto \sum_{k=0}^{N-1} \left(\frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} \psi_j e^{i2\pi jk/N} \right) \ket{k},
\end{equation*}
$$

where the coefficients

$$
\begin{equation*}
  \tilde{\psi_k} = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} e^{i2\pi jk/N} \psi_j
\end{equation*}
$$

are the the discrete Fourier transformof the probability amplitude vector $(\psi_0,\dots,\psi_j)$. The QFT defines an operator $\hat{F} : {}^\P \mathcal{H}^{\otimes n} \to {}^\P \mathcal{H}^{\otimes n}$ given by

$$
\begin{equation*}
  \hat{F} := \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} e^{i2\pi xy/N} \ket{x} \bra{y},
\end{equation*}
$$

where $\ket{x}, \ket{y} \in {}^\P \mathcal{H}^{\otimes n}$ are computational basis states. The Hermitian adjoint of $\hat{F}$ is

$$
\begin{equation*}
  \hat{F}^\dagger := \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} e^{-i2\pi xy/N} \ket{y} \bra{x}
\end{equation*}
$$

The operator $\hat{F}$ is unitary, which can be shown by computing the product $\hat{F} \hat{F}^\dagger$:

$$
\begin{align*}
  \hat{F} \hat{F}^\dagger =& \left(\frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} e^{i2\pi xy/N} \ket{x} \bra{y}\right) \left(\frac{1}{\sqrt{N}} \sum_{x'=0}^{N-1} \sum_{y'=0}^{N-1} e^{-i2\pi x'y'/N} \ket{y'} \bra{x'}\right) \\
  =& \frac{1}{N} \sum_{x,x',y,y'=0}^{N-1} e^{i2\pi (xy - x'y')/N} \ket{x} \underbrace{\braket{y|y'}}_{\delta_{yy'}}\bra{x'} \\
  =& \frac{1}{N} \sum_{x,x',y=0}^{N-1} \underbrace{e^{i 2\pi y(x - x')/N}}_{=\delta_{xx'}} \ket{x} \bra{x'} \\
  =& \frac{1}{N} \sum_{x,y=0}^{N-1} \ket{x}\bra{x} = \sum_{x=0}^{N-1} \ket{x}\bra{x} = \hat{I}_N,
\end{align*}
$$

where $\delta_{ij}$ is the Kronecker delta function and $\hat{I}_N$ is the identity operator of ${}^\P \mathcal{H}^{\otimes n}$.

### Tensor Product Representation

To explicitly compute the quantum Fourier transform of a $n$-qubit computational basis state 

$$
\begin{equation*}
  \ket{x} = \ket{x_{n-1} \dots x_0} = \bigotimes_{j=0}^{n-1} \ket{x_j},\; x_j \in\set{0,1},
\end{equation*}
$$

we need to refactor $\hat{F}$ in terms of the computational basis $\set{\ket{0},\ket{1}}$. As shown in @book_scherer_2019 [pp. 240-241], we proceed by applying $\hat{F}$ to $\ket{x}$ and using the binary expansion $y = \sum_{j=0}^{n-1} y_j 2^j$:

$$
\begin{align*}
  \hat{F}\ket{x} =& \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} \omega_N^{xy} \ket{y} = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} \omega_N^{x \sum_{j=0}^{n-1} y_j 2^j} \ket{y_{n-1}\dots y_0} \\
  =& \frac{1}{\sqrt{N}} \sum_{y_0 \dots y_{n-1} \in\set{0,1}} \prod_{j=0}^{n-1} \omega_N^{x y_j 2^j} \bigotimes_{k=n-1}^0 \ket{y_k} \\
  =& \frac{1}{\sqrt{N}} \sum_{y_0 \dots y_{n-1} \in\set{0,1}} \bigotimes_{k=n-1}^0 \omega_N^{x y_k 2^k} \ket{y_k} \\
  =& \frac{1}{\sqrt{N}} \bigotimes_{k=n-1}^0 \sum_{y_k \in\set{0,1}} \omega_N^{x y_k 2^k} \ket{y_k} \\
  =& \frac{1}{\sqrt{N}} \bigotimes_{k=n-1}^0 \left[\ket{0} + \omega_N^{x 2^k} \ket{1} \right],
\end{align*}
$$

where $\omega_N = e^{i2\pi/N}$. Introducing the binary fraction notation

$$
\begin{equation*}
  0_{.x_m \dots x_n} = \frac{x_m}{2} + \frac{x_{m+1}}{4} +\cdots+ \frac{x_n}{2^{n - m + 1}} = \sum_{j=m}^n x_j 2^{-(j-m+1)}, 
\end{equation*}
$$

and expanding the binary $x = \sum_{l=0}^{n-1} x_l 2^l$, we can rewrite $\omega_N^{x 2^k}$ as

$$
\begin{align*}
  \omega_N^{x 2^k} =& \exp\left(i\frac{2\pi}{2^n} x2^k \right) = \exp\left(i 2\pi \sum_{l}^{n-1} x_l 2^{l + k - n} \right) \\
  =& \exp\Biggl(i 2\pi \Biggl[\underbrace{\sum_{l=0}^{n-k-1} x_l 2^{l + k - n}}_{\in\mathbb{Q}} + \underbrace{\sum_{l=n-k}^{n-1} x_l 2^{l + k - n}}_{\in\mathbb{N}} \Biggr] \Biggr) \\
  =& \exp\left(i 2\pi\sum_{l=0}^{n-k-1} x_l 2^{l + k - n} \right) = e^{i2\pi 0_{.x_{n-1-k} \dots x_0}}
\end{align*}
$$

Inserting and rearranging the summation index yield

$$
\begin{align*}
  \hat{F}\ket{x} =& \frac{1}{\sqrt{N}} \bigotimes_{k=n-1}^0 \left[\ket{0} + e^{i2\pi 0_{.x_{n-1-k}\dots x_0}} \right] \\
  =& \frac{1}{\sqrt{N}} \bigotimes_{j=0}^{n-1} \left[\ket{0} + e^{i2\pi 0_{.x_j \dots x_0}} \right]
\end{align*}
$$

In particular, for the zero qubit $\ket{0}^{\otimes n}$, all binary fractions vanish, i.e. $0_{.x_j \dots x_0} = 0$. The quantum Fourier transform of $\ket{0}^{\otimes n}$ is therefore

$$
\begin{equation*}
  \hat{F} \ket{0}^{\otimes n} = \frac{1}{\sqrt{N}} \bigotimes_{j=0}^{n-1} \left[\ket{0} + \ket{1} \right],
\end{equation*}
$$

which is a uniform superposition over all basis states.

### Matrix Representation

In the computational basis of ${}^\P \mathcal{H}^{\otimes n}$, the matrix representation of the QFT operator $\hat{F}$ is given by

$$
\begin{equation*}
  \mathbf{F} =  \frac{1}{\sqrt{N}} \begin{bmatrix}
    \omega_N^{0\cdot 0} & \omega_N^{0\cdot 1} & \cdots & \omega_N^{0\cdot (N-1)} \\
    \omega_N^{1 \cdot 0} & \omega_N^{1\cdot 1} & \cdots & \omega_N^{1\cdot(N-1)} \\
    \vdots & \vdots & \ddots & \vdots \\
    \omega_N^{(N-1)\cdot 0}1 & \omega_N^{(N-1)\cdot 1} & \cdots & \omega_N^{(N - 1)\cdot(N - 1)}
  \end{bmatrix} = \frac{1}{\sqrt{N}} \begin{bmatrix}
    1 & 1 & \cdots & 1 \\
    1 & \omega_N & \cdots & \omega_N^{N-1} \\
    \vdots & \vdots & \ddots & \vdots \\
    1 & \omega_N^{N-1} & \cdots & \omega_N^{(N - 1)^2}
  \end{bmatrix}
\end{equation*}
$$

#### One-Qubit System
For a one-qubit system, with $N = 2$ and $\omega_2 = e^{i\pi} = -1$, the QFT matrix reduces to

$$
\begin{equation*}
  \mathbf{F}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix},
\end{equation*}
$$

which is precisely the Hadamard gate.

#### Two-Qubit System
For a two-qubit system, with $N = 4$ and $\omega_4 = e^{i \pi/2} = i$, the QFT matrix becomes

$$
\begin{equation*}
  \mathbf{F}_2 = \frac{1}{2} \begin{bsmallmatrix} 
    1 & 1 & 1 & 1 \\
    1 & i^1 & i^2 & i^3 \\
    1 & i^2 & i^4 & i^6 \\
    1 & i^3 & i^6 & i^9
  \end{bsmallmatrix} = \frac{1}{2} \begin{bsmallmatrix} 
    1 & 1 & 1 & 1 \\
    1 & i & -1 & -i \\
    1 & -1 & 1 & -i \\
    1 & -i & -1 & i
  \end{bsmallmatrix}
\end{equation*}
$$

#### Three-Qubit System
For a three-qubit system, with $N = 8$ and $\omega_8 = e^{i\pi/4} = (1 + i)/\sqrt{2}$, the QFT matrix is

$$
\begin{align*}
  \mathbf{F}_3 =& \frac{1}{\sqrt{8}} \begin{bsmallmatrix} 
    1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
    1 & e^{i\pi/4} & e^{i 2\pi/4} & e^{i 3\pi/4} & e^{i 4\pi/4}  & e^{i 5\pi/4} & e^{i 6\pi/4} & e^{i 7\pi/4} \\
    1 & e^{i 2\pi/4} & e^{i 4\pi/4} & e^{i 6\pi/4} & e^{i 8\pi/4} & e^{i 10\pi/4} & e^{i 12\pi/4} & e^{i14\pi/4} \\
    1 & e^{i 3\pi/4} & e^{i 6\pi/4} & e^{i 9\pi/4} & e^{i 12\pi/4} & e^{i 15\pi/4} & e^{i 18\pi/4} & e^{i 21\pi/4} \\
    1 & e^{i 4\pi/4} & e^{i 8\pi/4} & e^{i 12\pi/4} & e^{i 16\pi/4} & e^{i 20\pi/4} & e^{i 24\pi/4} & e^{i 28\pi/4}  \\
    1 & e^{i 5\pi/4} & e^{i 10\pi/4} & e^{i 15\pi/4} & e^{i 20\pi/4} & e^{i 25\pi/4} & e^{i 30\pi/4} & e^{i 35\pi/4} \\
    1 & e^{i 6\pi/4} & e^{i 12\pi/4} & e^{i 18\pi/4} & e^{i 24\pi/4} & e^{i 30\pi/4} & e^{i 36\pi/4}  & e^{i 42\pi/4} \\
    1 & e^{i 7\pi/4} & e^{i 14\pi/4} & e^{i 21\pi/4} & e^{i 28\pi/4} & e^{i 35\pi/4} & e^{i 42/\pi/4} & e^{i 49\pi/4}
  \end{bsmallmatrix} \\ 
  =& \frac{1}{\sqrt{8}} \begin{bsmallmatrix} 
    1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
    1 & e^{i\pi/4} & e^{i\pi/2} & e^{i 3\pi/4} & e^{i\pi} & e^{i 5\pi/4} & e^{i 3\pi/2} & e^{i 7\pi/4} \\
    1 & e^{i\pi/2} & e^{i\pi} & e^{i 3\pi/2} & e^{i 2\pi} & e^{i\pi/2} & e^{i \pi} & e^{i 3\pi/2} \\
    1 & e^{i 3\pi/4} & e^{i 3\pi/2} & e^{i \pi/4} & e^{i\pi} & e^{i 7\pi/4} & e^{i \pi/2} & e^{i 5\pi/4} \\
    1 & e^{i \pi} & e^{i 2\pi} & e^{i\pi} & e^{i 2\pi} & e^{i\pi} & e^{i 2\pi} & e^{i\pi} \\
    1 & e^{i 5\pi/4} & e^{i\pi/2} & e^{i 7\pi/4} & e^{i\pi} & e^{i \pi/4} & e^{i 3\pi/2} & e^{i 3\pi/4} \\
    1 & e^{i 3\pi/2} & e^{i\pi} & e^{i\pi/2} & e^{i 2\pi} & e^{i 3\pi/2} & e^{i\pi} & e^{i\pi/2} \\
    1 & e^{i 7\pi/4} & e^{i 3\pi/2} & e^{i 5\pi/4} & e^{i\pi} & e^{i 3\pi/4} & e^{i\pi/2} & e^{i \pi/4}
  \end{bsmallmatrix} = \frac{\sqrt{2}}{4} \begin{bsmallmatrix} 
    1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
    1 & \frac{1 + i}{\sqrt{2}} & i & \frac{i - 1}{\sqrt{2}} & -1 & -\frac{1 + i}{\sqrt{2}} & -i & \frac{1 - i}{\sqrt{2}} \\
    1 & i & -1 & -i & 1 & i & -1 & -i \\
    1 & \frac{i - 1}{\sqrt{2}} & -i & \frac{1 + i}{\sqrt{2}} & -1 & \frac{1 - i}{\sqrt{2}} & i & -\frac{1 + i}{\sqrt{2}} \\
    1 & -1 & 1 & -1 & 1 & -1 & 1 & -1 \\
    1 & -\frac{1 + i}{\sqrt{2}} & i & \frac{1 - i}{\sqrt{2}} & -1 & \frac{1 + i}{\sqrt{2}} & -i & \frac{i - 1}{\sqrt{2}} \\
    1 & -i & -1 & i & 1 & -i & -1 & i \\
    1 & \frac{1 - i}{\sqrt{2}} & -i & -\frac{1 + i}{\sqrt{2}} & -1 & \frac{i - 1}{\sqrt{2}} & i & \frac{1 + i}{\sqrt{2}}
  \end{bsmallmatrix}
\end{align*}
$$

### Quantum Circuit Implementation

As outlined in @book_scherer_2019 [pp. 241-245], the quantum Fourier transform can be realised using a sequence of Hadamard gates and controlled discrete phase shift gates. We start by deriving a general expression for the action of the Hadamard gate on a computational basis state. Applying the Hadamard gate to the computational basis states $\ket{0}$ and $\ket{1}$ produces the superpositions:

$$
\begin{equation*}
  \hat{H}\ket{0} = \frac{\ket{0} + \ket{1}}{\sqrt{2}},\quad \hat{H}\ket{1} = \frac{\ket{0} - \ket{1}}{\sqrt{2}}.
\end{equation*}
$$

This can be generalised using the identity $e^{i\pi} = -1$, so that for $x_j \in\set{0,1}$:

$$
\begin{equation*}
  \hat{H}\ket{x_j} = \frac{\ket{0} + e^{i\pi x_j} \ket{1}}{\sqrt{2}}
\end{equation*}
$$

In terms of the binary fraction $0_{.x_j} = x_j/2$, this becomes

$$
\begin{equation*}
  \hat{H} = \frac{\ket{0} + e^{i 2\pi 0_{.x_j}} \ket{1}}{\sqrt{2}}.
\end{equation*}
$$

Now consider an $n$-qubit register in the state $\ket{x} = \ket{x_{n-1} \dots x_0}$. Applying the Hadamard gate to the most significant qubit $\ket{x_{n-1}}$ yields

$$
\begin{equation*}
  \hat{H}_{n-1} \ket{x} = \frac{\ket{0} + e^{i 2\pi 0_{.x_{n-1}}} \ket{1}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0}.
\end{equation*}
$$

To accumulate the desired relative phases for the QFT, we apply a sequence of controlled phase shift gates. A controlled phase gate $\hat{R}_{jk} = \operatorname{diag}(1,1,1,e^{i\pi/2^{j-k}})$, where qubit $k$ is the control and qubit $j$ is the target, introduces a relative phase $e^{i\pi/2^{j-k}}$ to the $\ket{1}$ component of the target qubit if the control qubit is in state $\ket{1}$. Applying $\hat{R}_{n-1,n-2}$ to $\hat{H}_{n-1}\ket{x}$ gives

$$
\begin{align*}
  \hat{R}_{n-1,n-2} \hat{H}_{n-1} \ket{x} =& \frac{\ket{0} + \exp\left(i 2\pi 0_{.x_{n-1}} + i\pi \frac{x_{n-2}}{2}\right) \ket{1}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0} \\
  =& \frac{\ket{0} + e^{i 2\pi 0_{.x_{n-1} x_{n-2}}}\ket{1}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0},
\end{align*}
$$

Applying the remaining controlled phase gates to $\ket{x_{n-1}}$ results in

$$
\begin{align*}
  \hat{R}_{n-1,0} \cdots \hat{R}_{n-1,n-2} \hat{H}_{n-1} \ket{x} = \frac{\ket{0} + e^{i 2\pi 0_{.x_{n-1} \dots x_0}} \ket{1}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0}.
\end{align*}
$$

Repeating this process for each remaining factor qubit in $\ket{x_{n-2} \dots x_0}$ produces the final state:

$$
\begin{equation*}
  \frac{1}{\sqrt{2^n}} \bigotimes_{k=n-1}^0 \left[\ket{0} + e^{i2\pi 0_{.x_k \dots x_0}} \ket{1} \right],
\end{equation*}
$$

which is the QFT of $\ket{x}$, but in reversed qubit order. To restore the natural ordering, we apply a global swap operator $\hat{S}^{(n)}$ that reverses the qubit register. The complete QFT operator is therefore:

$$
\label{equation-fourier-gate}
  \hat{F} = \hat{S}^{(n)} \prod_{j=0}^{n-1} \left(\left[\prod_{k=0}^{j-1} \hat{R}_{jk} \right] \hat{H}_j \right).
$$

The circuit diagram in [](#figure:fourier-circuit) illustrates this construction.

```{figure} figures/fourier_circuit.pdf
:label: figure:fourier-circuit
:align: center

Circuit diagram for the quantum Fourier transform, constructed using Hadamard gates, controlled discrete phase shift gates, and final swap gates to reverse qubit order.
```

To obtain the inverse QFT, we take the Hermitian adjoint of [](#equation-fourier-gate). Using the identity $(\hat{A}\hat{B})^\dagger = \hat{B}^\dagger \hat{A}^\dagger$ we get:

$$
\begin{align*}
  \hat{F}^\dagger =& \left( \hat{S}^{(n)} \prod_{j=0}^{n-1} \left(\left[\prod_{k=0}^{j-1} \hat{P}_{jk} \right] \hat{H}_j \right)\right)^\dagger \\
  =& \left(\prod_{j=0}^{n-1} \left[\prod_{k=0}^{j-1} \hat{P}_{jk} \right] \hat{H}_j \right)^\dagger (\hat{S}^{(n)})^\dagger \\
  =& \prod_{j=n-1}^0 \left(\hat{H} \left[\prod_{k=j-1}^0 \hat{P}_{jk}^\dagger \right] \right) \hat{S}^{(n)}.
\end{align*}
$$

In the last step we used the Hermiticity of $\hat{H}$ and $\hat{S}$. Since $\hat{R}_{jk}^\dagger = \hat{R}_{jk}^{-1}$, the inverse QFT is constructed by reversing the gate sequence and negating the phases in the controlled phase gates. 

For an $n$ qubit register, the quantum circuit implementation of the QFT requires $O(n^2)$ gate operations, where $O$ is the big-O notation for asymptotic upper bound [@book_scherer_2019, pg. 245]. In comparison, the classicaal fast Fourier transform (FFT), which computes the discrete Fourier transform, requires $O(n2^n)$ operations for an input vector of length $2^n$ [@book_choi_2022, pg. 150]. This difference corresponds to an exponential speedup for for quantum algorithms leveraging QFT compared to classical FFT-based methods. 

However, while the FFT directly outputs the Fourier coefficients, the QFT produces a quantum state where these coefficients are encoded in the probability amplitudes of the qubit register. Due to the probabilistic nature of quantum measurements, extracting the Fourier coefficients typically requires multiple runs of the QFT and additional post-processing. This measurement overhead can complicate practical applications of the QFT.

## Quantum Phase Estimation

Quantum phase estimation (QPE) is an algorithm to estimate the phase $\phi\in [0,1)$ in the eigenvalue $e^{i2\pi\phi}$ of a unitary operator $\hat{U}\in\mathcal{U}({}^\P \mathcal{H}^{\otimes m}$. Since the eigenvalues of unitary operators have unit modulus and lie on the complex unit circle, they are characterised by their phase. If $\ket{u}\in\mathcal{H}$ is an eigenvector of $\hat{U}$, then it satisfies the eigenvalue equation

$$
\begin{equation*}
  \hat{U}\ket{u} = e^{i2\pi\phi} \ket{u}
\end{equation*}
$$

The QPE algorithm uses two qubit registers initialised in the state $\ket{\Psi_0} = \ket{0}^{\otimes n} \otimes\ket{u}$ [@book_nielsen_chuang_2000, pg. 221]:
1. The first register consists of $n\in\N_+$ qubits initialised in the zero state $\ket{0}^{\otimes n}$. The number of qubits $n$ determines the precision of the phase estimate and the probability for success.
2. The second register holds the eigenvector $\ket{u}$ of $\hat{U}$. If $\hat{U}$ is represented by an $2^m \times 2^m$-matrix, then $\ket{u}$ belongs to an $2^m$-dimensional Hilbert space, requiring $m$ qubits

As illustrated in [](#figure:quantum-phase-estimation), the phase estimation is performed in four steps. We first apply Hadamard gates $\hat{H}^{\otimes n}$ to each qubit in the first register $\ket{0}^{\otimes n}$, creating a uniform superposition

$$
\begin{align*}
  \ket{\Psi_1} =& (\hat{H}^{\otimes n} \otimes \hat{I}_m)\ket{\Psi_0} = \frac{1}{\sqrt{2^n}} (\ket{0} + \ket{1})^{\otimes n} \otimes \ket{u} = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} \ket{k} \otimes\ket{u}.
\end{align*}
$$

This step entangles the computational basis states $\ket{k}$ in the first register with the eigenstate $\ket{u}$ in the second.

```{figure} figures/quantum_phase_estimation.pdf
:label: figure:quantum-phase-estimation
:align: center

Circuit diagram for the quantum phase estimation algorithm. The circuit comprises an initial layer of Hadamard gates, followed by a sequence of controlled binary powers of a unitary operator $\hat{U}^{2^i}$, and concludes with the inverse quantum Fourier transform $\hat{F}^\dagger$ to extract the phase information.
```

Next, we apply a sequence of controlled unitary gates that implement $\hat{U}^{2^k}$ conditioned on the $k$-th qubit of the first register. The composition of controlled unitary gates can be expressed in the form

$$
\begin{equation*}
  \hat{U}_\text{c} = \sum_{k=0}^{2^n - 1} \ket{k}\bra{k} \otimes \ket{U}^k .
\end{equation*}
$$

Applying $\hat{U}_c$ to $\ket{\Psi_1}$ and using that $\hat{U}^k \ket{u} = e^{i2\pi\phi k} \ket{u}$, we obtain the state 

$$
\begin{align*}
  \ket{\Psi_2} =& \hat{U}_\text{c} \ket{\Psi_1} = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} \ket{k} \otimes \hat{U}^k \ket{u} \\
  =& \left(\frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} e^{i2\pi\phi k} \right) \otimes\ket{u} \\
  =& \frac{1}{\sqrt{2^n}} \left(\bigotimes_{k=0}^{2^n - 1} \ket{0} + e^{i2\pi 0_{.\phi_1,\dots,\phi_k}} \right) .
\end{align*}
$$

From this point on, the second register remains unchanged. For convenience we write $\ket{\Psi_2} = \ket{\tilde{\Psi_2}} \otimes\ket{u}$, where the state of the first qubit register $\ket{\tilde{\Psi_2}}$ is the only we need to consider for the rest of the algorithm.

In the third step, we apply the inverse quantum Fourier transform $\hat{F}_{2^n}^\dagger$ to the first qubit register:

$$
\begin{align*}
  \ket{\tilde{\Psi_3}} =& \hat{F}^\dagger \ket{\tilde{\Psi_2}} \\
  =& \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} e^{i2\pi\phi k} \left(\frac{1}{\sqrt{2}} \sum_{j=0}^{2^n - 1} e^{-i2\pi kj/2^n} \ket{j} \right) \\
  =& \frac{1}{2^n} \sum_{j,k=0}^{2^n - 1} e^{i 2\pi k (2^n \phi - j)/2^n} \ket{j} .
\end{align*}
$$

Decomposing the state in the computational basis as $\ket{\tilde{\Psi_3}} = \sum_{j=0}^{2^n - 1} c_j \ket{j}$, the probability amplitudes are given by

$$
\label{equation-qpe-coefficients}
  c_j := \frac{1}{2^n} \sum_{k=0}^{2^n - 1} e^{i2\pi k (2^n \phi - j)/2^n} \ket{j} .
$$

We can split $\phi$ into its binary components by writing $2^n \phi = b + r$, where $b\in\Z$ with $0 \leq b < 2^n$ and $r \in[0,1)$. Here $b$ is the integer such that $b/2^n = 0_{.b_{n-1},\dots,b_0}$ is the best $n$ bit approximation of $\phi$. Substituting this into [](#equation-qpe-coefficients) yields

$$
\begin{align*}
  c_j =& \frac{1}{2^n} \sum_{k=0}^{2^n - 1} e^{i2\pi k (b + r - j)/2^n} \\
  =& \frac{1}{2^n} \sum_{k=0}^{2^n - 1} e^{-i2\pi k (j - b)/2^n} e^{i2\pi r k/2^n} .
\end{align*}
$$

In particular, if $r = 0 \iff \phi = b/2^n$ then by the orthogonality relation the probability amplitudes become

$$
\begin{equation*}
  c_j = \frac{1}{2^n} \sum_{k=0}^{2^n - 1} e^{-i2\pi k (j - b)/2^n} = \delta_{j b} .
\end{equation*}
$$

Since $\phi$ in this case is given by an $n$-bit binary fraction

$$
\begin{equation*}
  \phi = \sum_{k=n-1}^0 \phi_k 2^{-(n - k + 1)},
\end{equation*}
$$

the inverse quantum Fourier transform yields $\ket{\tilde{\Psi}_3} = \ket{2^n \phi}$ with certainty. Thus, measuring $\ket{\tilde{\Psi}_3}$ in the computational basis returns $\ket{\phi_{n-1},\dots,\phi_0}$, the $n$-bit binary representation of $\phi$, with probability $1$.

The final step involves measuring the first register $\ket{\tilde{\Psi_3}}$ in the computational basis. This yields an outcome $\ket{y}$ with probability

$$
\begin{align*}
  \Pr(y) :=& |c_y|^2 = \left|\frac{1}{2^n} \sum_{k=0}^{2^n - 1} e^{-i2\pi k (y - 2^n \phi)/2^n} \right|^2 \\
  =& \frac{1}{2^{2n}} \left| \frac{1 - e^{i2\pi (y - 2^n \phi)}}{1 - e^{i2\pi (y - 2^n \phi)/2^n}} \right|^2 .
\end{align*}
$$

As shown in [](#figure:phase-probability), the probability distribution $\Pr(y)$ sharply peaks around $2^n \phi$ for increasing $n$. The probability that the estimation error $\epsilon := |y/2^{n} - \phi|$ is less than $2^{-n}$ is bounded below by [@book_choi_2022, pg. 165],

$$
\begin{equation*}
  \Pr(\epsilon < 2^{-n}) \geq \frac{8}{\pi^2} \approx 0.81 .
\end{equation*}
$$

To resolve the phase $\phi$ with a target error $\epsilon = 2^{-n}$, we require $n$ precision qubits such that

$$
\begin{equation*}
  \frac{1}{2^n} \leq\epsilon \implies n \geq \log_2 (1/\epsilon).
\end{equation*}
$$

Thus, the number of precision qubits scales as $O(\log_2 (1/\epsilon))$. The $j$th precision qubit controls an application of the unitary operator $\hat{U}^{2^j}$, which requires $2^j$ sequential applications of $\hat{U}$ in the quantum circuit [@article_tilly_etal_2022, pg. 14]. The total number of controlled-$\hat{U}$ operations across all precision qubits is therefore

$$
\begin{equation*}
  \sum_{j=0}^{n-1} 2^j = 2^n - 1
\end{equation*}
$$

which scales exponentially as $O(2^n) = O(1/\epsilon)$.

```{figure} figures/phase_probability.pdf
:label: figure:phase-probability
:align: center

The probability distribution $\Pr(y)$ of measuring outcome $y$ in the quantum phase estimation alogrithm, interpreted as estimating the phase $\phi \approx y/2^n$. The plot shows results for $n \in\set{3,4,8}$ precision qubits, with true phase $\phi = 17/31$.
```

### Energy Estimation

The QPE algorithm can be applied to estimate the eigenvalues of a Hamiltonian $\hat{H}$. To do so, the Hamiltonian is embedded in a unitary operator $\hat{U}$ via time evolution [@article_tilly_etal_2022, pg. 14]:

$$
\begin{equation*}
  \hat{U} = e^{i\hat{H}t},
\end{equation*}
$$

where the time parameter $t\in\R$ serves as a scaling factor. If $\ket{\psi_j}$ is an eigenstate of $\hat{H}$ with eigenvalue $E_j$, then it is also an eigenstate of $\hat{U}$ with eigenvalue $e^{iE_j t}$. Thus, by applying QPE to $\hat{U}$, we can obtain an estimate of the corresponding phase $\phi_j$ such that $e^{iE_j t} = e^{i2\pi\phi_j}$. Solving for the energy eigenvalue yields

$$
\begin{equation*}
  E_j = \frac{2\pi\phi_j}{t}
\end{equation*}
$$

#### One-Qubit Hamiltonian

An interacting one-qubit system can be modelled by a real symmetric Hamiltonian of the form $\hat{H}_1 = \hat{H}_0 + \hat{H}_\text{I} \in\R^{2\times 2}$, where the stationary component is diagonal,

$$
\begin{equation*}
  \hat{H}_0 = \begin{bmatrix} E_1 & 0 \\ 0 & E_2 \end{bmatrix},
\end{equation*}
$$

and the interaction component is symmetric,

$$
\begin{equation*}
  \hat{H}_\text{I} = \begin{bmatrix} V_{11} & V_{12} \\ V_{21} & V_{22} \end{bmatrix}.
\end{equation*}
$$

In the computational basis $\set{\ket{0}, \ket{1}}$, the stationary Hamiltonian satisfies the eigenvalue equations

$$
\begin{equation*}
  \hat{H}_0 \ket{0} = E_1 \ket{0},\; \hat{H}_0 \ket{1} = E_2 \ket{1}.
\end{equation*}
$$

Using Pauli matrices, the stationary term can be rewritten as

$$
\begin{equation*}
  \hat{H}_0 = \mathcal{E}\hat{I}_2 + \Omega \hat{\sigma}_z,\; \mathcal{E} = \frac{E_1 + E_2}{2},\; \Omega = \frac{E_1 - E_2}{2},
\end{equation*}
$$

and similarly, the interaction term becomes

$$
\begin{equation*}
  \hat{H}_I = c\hat{I}_2 + \omega_z \hat{\sigma}_z + \omega_x \hat{\sigma_x},
\end{equation*}
$$

with parameters 

$$
\begin{equation*}
  c = \frac{V_{11} + V_{22}}{2},\; \omega_z = \frac{V_11 - V_22}{2},\; \omega_x = V_{12} = V_{21}.
\end{equation*}
$$

To study the effect of the interactions, we introduce a parameter $\lambda\in[0,1]$ and define the full Hamiltonian as

$$
\begin{equation*}
  \hat{H}_1 = \hat{H}_0 + \lambda\hat{H}_\text{I}.
\end{equation*}
$$

The limits $\lambda = 0$ and $\lambda = 1$ correspond to the non-interacting and fully interacting systems, respectively. The eigenvalues of $\hat{H}_1$ are given by (see [](#eigenvalues-of-one-qubit-hamiltonian) for derivation)

$$
\begin{equation*}
  E_\pm = \mathcal{E} + \lambda c \pm \sqrt{(\Omega + \lambda \omega_z)^2 + (\lambda\omega_x)^2}.
\end{equation*}
$$

#### Two-Qubit Hamiltonian

An interacting one-qubit system can be modelled by a real symmetric Hamiltonian of the form $\hat{H}_2 = \hat{H}_0 + \hat{H}_\text{I} \in\R^{4\times 4}$, where the stationary component is given by

$$
\begin{equation*}
  \hat{H}_0 = \operatorname{diag}(\epsilon_{00}, \epsilon_{10}, \epsilon_{01}, \epsilon_{11})
\end{equation*}
$$

while the interaction component is

$$
\begin{equation*}
  \hat{H}_I = H_x \hat{\sigma}_x \otimes \hat{\sigma}_x + H_z \hat{\sigma}_z \otimes \hat{\sigma_z},
\end{equation*}
$$

The parameters $H_x, H_z \in\R$ describe the strength of the interactions along the $x$- and $z$-axis, respectively. In the computational basis $\set{\ket{00}, \ket{01}, \ket{10}, \ket{11}}$, the full Hamiltonian matrix is given by

$$
\begin{equation*}
  \hat{H}_2 = \left[\begin{smallmatrix}
    \epsilon_{00} + H_z & 0 & 0 & H_x \\
    0 & \epsilon_{01} - H_z & H_x & 0 \\
    0 & H_x & \epsilon_{10} - H_z & 0 \\
    H_x & 0 & 0 & \epsilon_{11} + H_z
  \end{smallmatrix}\right]
\end{equation*}
$$

# Results

## Precision of Quantum Phase Estimation

[](#figure:qpe-measurements) shows simulated measurement outcomes from quantum phase estimation for a target phase of $\phi = 17/31$, using $4$ precision qubits. As expected, the resulting distribution is sharply peaked around the true phase. At this resolution, the phase estimation is expected to lie within an error of $1/2^4 = \num{6.25e-02}$ from the true value with approximately $81\%$ confidence. In this simulation, the observed success probability was about $91.7\%$.

```{figure} figures/qpe_measurements.pdf
:label: figure:qpe-measurements
:align: center

Measurement outcome distribution over $1000$ shots from quantum phase estimation for a target phase $\phi = 17/31$, using $4$ precision qubits.
```

## Quantum Phase Estimation of One-Qubit Hamiltonian

The energy levels of the one-qubit Hamiltonian $\hat{H}_1$ were computed using quantum phase estimation (QPE). The Hamiltonian parameters were set to

$$
\begin{equation*}
  E_1 = 0,\quad E_2 = 4,\quad V_{11} = −V_{22} = 3, \quad V_{12} = V_{21} = 0.2,
\end{equation*}
$$

defining a system with both diagonal and off-diagonal interactions.

The numerical experiment was conducted using $8$ precision qubits. At this resolution, the phase estimate $\tilde{\phi}$ lies within $1/2^8 \approx \num{3.91e-03}$ of the true phase with approximately $81\%$ confidence. As shown in [](#figure:qpe-unary-result) these phase estimates are sufficiently accurate to resolve the eigenvalue structure of $\hat{H}_1$, including the presence of an avoided crossing near $\lambda = 2/3$. This behaviour is typical for two-level quantum systems, where the eigenstates undergo a transition in their dominant character with increasing interaction strength. 

```{figure} figures/qpe_unary_result.pdf
:label: figure:qpe-unary-result
:align: center

A plot of energy estimates for a one-qubit Hamiltonian, obtained via quantum phase estimation using $8$ precision qubits.
```

In the non-interacting limit $\lambda = 0$, the Hamiltonian reduces to the diagonal form $\hat{H}_1 = \hat{H}_0$, with eigenstates corresponding to the computational basis states $\ket{0}$ and $\ket{1}$. As the interaction strength $\lambda$ increases, the off-diagonal coupling terms $V_{12} = V_{21}$ induce mixing between the computational basis states. Around $\lambda = 2/3$, the eigenstates become maximally mixed and they swap their dominant character: the lower eigenstate transition from being predominantly $\ket{0}$ to primarily $\ket{1}$, while the higher eigenstate undergoes the reverse transition. Beyond this avoided crossing point, $\lambda > 2/3$, the eigenstates continue to separate in energy, with their dominant character exchanged.

For comparison, [](#figure:vqe-unary-result) presents the results obtained by estimating the ground-state energies of $\hat{H}_1$ using the variational quantum eigensolver (VQE) algorithm. Similar to QPE, the VQE successfully captures the ground-state energy behaviour characteristic of a two-level quantum system.

As a variational optimisation method, the VQE is designed to approximate only the ground-state energy. In contrast, the QPE algorithm can estimate multiple eigenvalues, including excited states, provided that the input state has sufficient overlap with the corresponding eigenstate. The VQE, on the other hand, operates on random initial states that are iteratively refined through classical optimisation to converge toward the true ground state.

```{figure} figures/vqe_unary_result.pdf
:label: figure:vqe-unary-result
:align: center

A plot of VQE energies of $\braket{H_1 (\theta, \phi)}$, optimised using gradient descent with Adaptive Moment Estimation (ADAM) over a maximum of $500$ epochs. The expectation values $\braket{H_1}$ were computed from quantum circuit measurements over $1000$ shots.
```

## Quantum Phase Estimation of Two-Qubit Hamiltonian

The energy levels of the two-qubit Hamiltonian $\hat{H}_2$ were computed using quantum phase estimation (QPE). The Hamiltonian parameters were set to

$$
\begin{equation*}
  H_x = 2.0,\quad H_z = 3.0,\quad \epsilon_{00} = 0.0,\quad \epsilon_{01} = 2.5,\quad \epsilon_{10} = 6.5,\quad \epsilon_{11} = 7.0.
\end{equation*}
$$

[](#figure:qpe-binary-result) shows the QPE results obtained with $8$ precision qubits. At this resolution, the QPE algorithm successfully resolves the eigenvalue structure of this four-level qubit system, which exhibit two distinct characteristics. The two lowest energy states, corresponding to eigenvalues $E_1$ and $E_2$, display an avoided crossing around $\phi\approx 0.4$, resembling the beviour observed in the two-level system. In contrast, the the two upper energy states, corresponding to eigenvalues $E_3$ and $E_4$, diverge monotonically as $\lambda$ increases. This indicates that the higher-energy states are less affected by the interaction mechanism responsible for mixing in the lower states.

```{figure} figures/qpe_binary_result.pdf
:label: figure:qpe-binary-result
:align: center

A plot of energy estimates for a two-qubit Hamiltonian, obtained via quantum phase estimation using $8$ precision qubits.
```

For comparison, [](#figure:vqe-binary-result) presents the results using the variational quantum eigensolver (VQE) algorithm to estimate the ground-state energies of $\hat{H}_2$. The results demonstrate that the VQE accurately captures the ground-state energy behaviour of a four-level quantum system with non-degenerate eigenvalues.

```{figure} figures/vqe_binary_result.pdf
:label: figure:vqe-binary-result
:align: center

A plot of VQE energies of $\braket{H_2 (\theta, \phi)}$, optimised using gradient descent with Adaptive Moment Estimation (ADAM) over a maximum of $500$ epochs. The expectation values $\braket{H_2}$ were computed from quantum circuit measurements over $1000$ shots.
```

## Comparison of Quantum Phase Estimation and Variational Quantum Eigensolver

This section compares the use of the quantum phase estimation (QPE) and the variational quantum eigensolver (VQE) algorithms to estimate energy eigenvalues of a Hamiltonian (a description of the VQE is given in [](#variational-quantum-eigensolver-vqe)). [](#table:qpe-vqe-comparison) summarises the key features of these two techniques.

A major advantage of QPE is its ability to estimate both ground and excited energy levels, allowing it to resolve the full spectral structure of a quantum system. In contrast, VQE is generally limited to estimating only the ground-state energy level, making it less suitable for studying excited-state phenomena. However, QPE requires the input state to have significant overlap with the eigenstate corresponding to the desired eigenvalue [@article_tilly_etal_2022, pg. 14]. For complex systems, this often necessitates a separate procedure to prepare an suitable eigenstate approximation. VQE, by comparison, employs a parametrised ansatz state that is iteratively optimised to approximate the ground state, without requiring prior knowledge of it.

A quantum circuit implementation of QPE is inherently restricted to unitary operators acting on $2^n$-dimensional Hilbert spaces. Consequently, Hamiltonians defined on arbitrary finite-dimensional spaces must be embedded into a $2^n$-dimensional space via zero-padding or some other encoding scheme. This embedding can introduce additional overhead in circuit size and complexity. On the other hand, VQE can be applied to any finite-dimensional Hamiltonian, provided it can be expressed as a linear combination of Pauli string operators. This makes the VQE more flexible for a broader range of quantum systems, without requiring embedding.

Another advantage is that QPE can estimate eigenvalues to a target precision $\epsilon$ with constant success probability, requiring only $O(1)$ repetitions [@article_tilly_etal_2022, pg. 14]. Achieving the same level of precision with VQE requires $O(1/\epsilon^2)$ measurement shots due to the statistical nature of expectation value estimation. Moreover, VQE depends on classical optimisation routines to minimise the energy expectation value, which introduces additional computational overhead. These optimisation procedures can also be sensitive to local minima, particularly in systems with nearly degenerate ground-state energy levels. This can prevent convergence to the true ground state.

A key limitation of QPE is that number of controlled unitary operations required to achieve a target precision $\epsilon$, scales exponentially as $O(1/\epsilon)$. This renders QPE impractical for large Hamiltonians on current or near-term quantum hardware. In contrast, VQE typically requires circuits with constant-depth scaling, $O(1)$, with respect to precision. This makes VQE more scalable for larger and more complex quantum systems.

:::{table} Comparison of key features of the quantum phase estimation (QPE) and variational quantum eigensolver (VQE) algorithms. Here $\epsilon$ denotes the target precision for estimating energy eigenvalues.
:label: table:qpe-vqe-comparison
:align: center

| Feature                       | QPE                                                       | VQE                                      |
| ----------------------------- | --------------------------------------------------------- | ---------------------------------------- |
| **Target eigenvalues**        | Ground and excited states                                 | Ground states only                       |
| **Input requirements**        | Approximate eigenstate                                    | Parametrised ansatz statae               |
| **Hamiltonian compatibility** | Requires embedding into a $2^n$-dimensional Hilbert space | Must be decomposable into Pauli strings  |
| **Precision scaling**         | $O(1/\epsilon)$ gate complexity                           | $O(1/\epsilon^2)$ measurement complexity |
| **Classical post-processing** | Minor (bitstring decoding)                                | Requires iterative optimisation          |
:::

# Conclusion

This project investigated the application of the quantum Fourier transform (QFT) within the quantum phase estimation (QPE) algorithm to estimate energy eigenvalues of a quantum systems. By applying QPE on one- and two-qubit Hamiltonians, we demonstrated its capability to resolve both ground and excited state energies with high precision. The results were benchmarked against the variational quantum eigensolver (VQE), highlighting key differences in algorithmic approach and performance.

In theory, QPE provides a powerful framework for spectral analysis of quantum systems, especially when high-precision estimates of multiple eigenvalues are required. However, its practical deployment is constrained by two major challenges: the need for an input state with significant overlap with a true eigenstate, and the exponential growth in circuit depth due to repeated controlled unitary operations. By contrast, the VQE is restricted to estimating ground-state energies and depends on classical optimisation, which may suffer from convergence issues. Nonetheless, its relatively shallow circuit depth makes the VQE more suitable for implementation on current and near-term quantum hardware.

# Appendix

## Variational Quantum Eigensolver (VQE)

The variational quantum eigensolver (VQE) is a hybrid classical-quantum algorithm designed to estimate the ground state energy of a Hamiltonian. This description of the algorithm is based on @article_tilly_etal_2022.

The energy levels of a quantum system $\mathcal{H}$ with Hamiltonian $\hat{H}$ are given by the time-independent Schrödinger equation

$$
\begin{equation*}
  \hat{H}\ket{\psi} = E \ket{\psi}.
\end{equation*}
$$

Since the Hamiltonian is Hermitian, it has a complete orthonormal eigenbasis $\set{\ket{q}_j}_{j=0}^n$, meaning that any state $\ket{\psi}\in\mathcal{H}$ can be expanded as

$$
\begin{equation*}
  \ket{\psi} = \sum_{j=1}^n c_j \ket{\varphi_j}.
\end{equation*}
$$

If $\hat{H}$ has non-degenerate eigenvalues $\set{E_j}_{j=1}^n$, the expectation value of $\hat{H}$ in terms of $\ket{\psi}$ is given by

$$
\begin{align*}
  \braket{\hat{H}}_\psi = \braket{\psi|\hat{H}|\psi} =& \sum_{j,k=1}^n c_j^* c_k \braket{\varphi_j|\hat{H}|\varphi_k} \\
  =& \sum_{j,k=1}^n c_j^* c_k E_j \overbrace{\braket{\varphi_j|\varphi_k}}^{=\delta_{jk}} \\
 =& \sum_{j=1}^n |c_j|^2 E_j \geq E_0 \overbrace{\sum_{j=1} |c_j|^2}^{=1} = E_0.
\end{align*}
$$

By the Rayleigh-Ritz variational principle, this expectation value provides an upper bound for the ground state energy $E_0$

$$
\begin{equation*}
  E_0 \leq \braket{\psi|\hat{H}|\psi} = \braket{\hat{H}}.
\end{equation*}
$$

The objective of the VQE is therefore to find a parametrisation of $\ket{\psi(\boldsymbol{\theta})}$, called an ansatz, that minimizes the expectation value $\braket{\hat{H}}_\psi$, where $\boldsymbol{\theta}\in\Theta$ is a vector of tunable parameters. To implement this on a quantum circuit, we express the ansatz in terms of a generic parametrised unitary operator $\hat{U}(\boldsymbol{\theta})$ applied to an initial state $\ket{\psi_0}$:

$$
\begin{equation*}
  \ket{\psi(\boldsymbol{\theta})} = \hat{U}(\boldsymbol{\theta}) \ket{\psi_0}.
\end{equation*}
$$

The VQE cost function is then defined as

$$
\begin{equation*}
  \braket{\hat{H}(\boldsymbol{\theta})} = \braket{\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})}.
\end{equation*}
$$

For a quantum circuit implementation, the Hamiltionian must be expressed as a weighted sum of Pauli operators $\hat{P}_A \in \set{\hat{I}_2, \hat{X}, \hat{Y}, \hat{Z}}^{\otimes N}$:

$$
\begin{equation*}
  \hat{H} = \sum_a^\mathcal{P} w_a \hat{P}_a.
\end{equation*}
$$

Substituting this into the cost function yields the minimisation problem

$$
\begin{equation*}
  E_\text{VQE} = \argmin_{\boldsymbol{\theta}} \sum_a^\mathcal{P} w_a \braket{\psi(\boldsymbol{\theta})|\hat{P}_a|\psi(\boldsymbol{\theta})}.
\end{equation*}
$$

Each term $E_{P_a} = \braket{\psi(\boldsymbol{\theta})|\hat{P}_a|\psi(\boldsymbol{\theta})}$ corresponds to the expectation value of a Pauli string $\hat{P}_a$, which can be measured on a quantum device, while the summation and parameter optimisation $E(\boldsymbol{\theta}) = \argmin_{\boldsymbol{\theta}} \sum_a w_a E_{P_a}$ are performed using a classical optimisation algorithm, making VQE a hybrid quantum-classical method.

### Measurements in The Computational Basis

Quantum circuits measure in the $\hat{Z}$-basis by default, so we express all Pauli operators in terms of $\hat{Z}$ using the basis transformations

$$
\begin{equation*}
  \hat{Z} = \hat{HXH},\quad \hat{Z} = \hat{H}\hat{S}^\dagger \hat{Y}\hat{H}\hat{S}^\dagger,
\end{equation*}
$$

where $\hat{S} = \hat{Z}^{1/2}$ is the S-gate.

In an $n$-qubit system, an arbitrary Pauli string $\hat{P}$ acting non-trivially on a subset $Q\subseteq \set{1,\dots,n}$ of can be transformed into a $\hat{Z}$-basis measurement using unitary rotations $\hat{R}_{\sigma_p}$ such that

$$
\begin{equation*}
  \bigotimes_{p\in Q} \hat{\sigma}_p = \left(\bigotimes_{p\in Q} \hat{R}_{\sigma_p}^\dagger \hat{Z} \hat{R}_{\sigma_p}\right).
\end{equation*}
$$

The expectation value of $\hat{P}$ is given by a linear combination of probabilities:

$$
\begin{equation*}
  \braket{\hat{P}}_\psi = \sum_{x\in\set{0,1}^n}(-1)^{\sum_{p\in Q} x_p} |\braket{x|\varphi}|^2,
\end{equation*}
$$

where $\ket{\varphi} = \left(\otimes_{p\in Q} \hat{R}_{\sigma_p} \right)\ket{\psi}$. The probability $\Pr(\ket{\varphi}\to\ket{x}) = |\braket{x|\varphi}|^2$ that the state $\ket{\varphi}$ collapses to the state $\ket{x}$ when measured is estimated statistically as

$$
\begin{equation*}
  \Pr(\ket{\varphi}\to\ket{x}) \approx \sum_{m=1}^M \frac{x_m}{M},
\end{equation*}
$$

where $M\in\N_+$ is the number of measurements. By the law of large numbers, the approximation of $\Pr(\ket{\varphi}\to\ket{x})$ converges to its true value as $M\to\infty$.

For precision $\epsilon$, each expectation subroutine within the VQE requires $O(1/\epsilon^2)$ samples from circuits with depth $O(1)$.

Given $M_0$ and $M_1$ as the number of $0$ and $1$ measurement outcomes, the expectation value of $\hat{P}_a$ is approximately

$$
\begin{equation*}
  \braket{P}_\psi \approx \frac{M_0 - M_1}{M}.
\end{equation*}
$$

### Gradient of the Expectation Value

To apply gradient descent methods to optimize the ansatz state $\ket{\psi(\boldsymbol{\theta})} = \hat{U}(\boldsymbol{\theta})\ket{\psi_0}$, we need to compute the gradient of $\braket{\hat{H}(\boldsymbol{\theta})}$. These gradients can be computed using the parameter shift rule:

$$
\begin{equation*}
  \frac{\partial}{\partial\theta_j} \braket{H(\boldsymbol{\theta})} = \frac{1}{2}\left(\hat{H}\left(\boldsymbol{\theta} + \frac{\pi}{2}\unitvec{e}_j \right) - \hat{H}\left(\boldsymbol{\theta} - \frac{\pi}{2}\unitvec{e}_j \right) \right).
\end{equation*}
$$

## Eigenvalues of One-Qubit Hamiltonian

To find the eigenvalues of the one-qubit Hamiltonian $\hat{H}_1$, we must solve the characteristic equation $\det(\hat{H}_1 - E\hat{I}) = 0$. The identity matrix $\hat{I}$ contributes a constant energy shift $\mathcal{E} + \lambda c$, meaning the relevant part of the Hamiltonian for eigenvalue calculation is

$$
\begin{equation*}
  \hat{H}' = (\Omega + \lambda\omega_z) \hat{Z} + \lambda\omega_x \hat{X},
\end{equation*}
$$

or in matrix form

$$
\begin{equation*}
  \hat{H}' = \begin{bmatrix} \Omega + \lambda\omega_z & \lambda \omega_x \\ \lambda\omega_x & -(\Omega + \lambda\omega_z) \end{bmatrix}
\end{equation*}
$$

To find the eigenvalues, we solve the characteristic equation $\det(\hat{H}' - E'\hat{I}) = 0$:

$$
\begin{align*}
  0 =& \begin{vmatrix} \Omega + \lambda\omega_z - E' & \lambda\omega_x \\ \lambda\omega_x & -(\Omega + \lambda\omega_z) - E' \end{vmatrix} \\
  =& (\Omega + \lambda\omega_z - E')(-(\Omega + \lambda\omega_z) - E') - (\lambda\omega_x)^2 \\
  =& -(\Omega + \lambda\omega_z - E')(\Omega + \lambda\omega_z + E') - (\lambda\omega_x)^2 \\
  =& -(\Omega + \lambda\omega_z)^2 + E'^2 - (\lambda\omega_x)^2
\end{align*}
$$

Rearranging yields

$$
\begin{equation*}
  E'^2 = (\Omega + \lambda\omega_z)^2 + (\lambda\omega_x)^2.
\end{equation*}
$$

Taking the square root, we find the reduced eigenvalues

$$
\begin{equation*}
  E'_\pm = \pm \sqrt{(\Omega + \lambda \omega_z)^2 + (\lambda\omega_x)^2}
\end{equation*}
$$

Adding the constant shift $\mathcal{E} + \lambda c$, we finally arrive at

$$
\begin{equation*}
  E_\pm = \mathcal{E} + \lambda c \pm \sqrt{(\Omega + \lambda \omega_z)^2 + (\lambda\omega_x)^2}
\end{equation*}
$$