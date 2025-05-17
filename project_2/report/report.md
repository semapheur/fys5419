---
title: FYS5419 - Project 2
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
  '\unitvec': '{\hat{\mathbf{#1}}}'
bibliography: references.bib
abstract: |
---

# Theory

## Quantum Fourier Transform

Let ${}^\P \mathcal{H}^{\otimes n}$ denote the $n$-qubit Hilbert space, with computational basis $\set{\ket{j}}_{j=0}^{N-1}$ for $N = 2^n$. The quantum Fourier transform (QFT) is a linear tranformation defined on computational basis states $\ket{j}\in {}^\P \mathcal{H}^{\otimes n}$ by

$$
  \ket{j} \mapsto \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{i2\pi jk/N} \ket{k}
$$

By linearity, the QFT maps an arbitrary qubit state $\ket{\psi} = \sum_{j=0}^{N-1} \psi_j \ket{j} \in {}^\P \mathcal{H}^{\otimes n}$ as

$$
  \ket{\psi} \mapsto \sum_{k=0}^{N-1} \left(\frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} \psi_j e^{i2\pi jk/N} \right) \ket{k},
$$

where the coefficients

$$
  \tilde{\psi_k} = \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} e^{i2\pi jk/N} \psi_j
$$

are the the discrete Fourier transformation of the probability amplitude vector $(\psi_0,\dots,\psi_j)$. The QFT defines an operator $\hat{F} : {}^\P \mathcal{H}^{\otimes n} \to {}^\P \mathcal{H}^{\otimes n}$ given by

$$
  \hat{F} := \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} e^{i2\pi xy/N} \ket{x} \bra{y},
$$

where $\ket{x}, \ket{y} \in {}^{\otimes n} \to {}^\P \mathcal{H}^{\otimes n}$ are computational basis states. The Hermitian adjoint of $\hat{F}$ is

$$
  \hat{F}^\dagger := \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} e^{-i2\pi xy/N} \ket{y} \bra{x}
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

where $\hat{I}_N$ is the identity operator of ${}^\P \mathcal{H}^{\otimes n}$.

### Tensor Product Representation

To explicitly compute the quantum Fourier transform of a $n$-qubit computational basis state 

$$
  \ket{x} = \ket{x_{n-1} \dots x_0} = \bigotimes_{j=0}^{n-1} \ket{x_j},\; x_j \in\set{0,1},
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
  0_{.x_m \dots x_n} = \frac{x_m}{2} + \frac{x_{m+1}}{4} +\cdots+ \frac{x_n}{2^{n - m + 1}} = \sum_{j=m}^n x_j 2^{-(j-m+1)}, 
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

In particular, for the zero qubit $\ket{0}^{\otimes n}$, all binary fractions vanish, i.e. $0_{.x_j \dots x_0} = 0$. The quantum Fourier transformation of $\ket{0}^{\otimes n}$ is therefore

$$
  \hat{F} \ket{0}^{\otimes n} = \frac{1}{\sqrt{N}} \bigotimes_{j=0}^{n-1} \left[\ket{0} + \ket{1} \right],
$$

which is a uniform superposition over all basis states.

### Matrix Representation

In the computational basis of ${}^\P \mathcal{H}^{\otimes n}$, the matrix representation of the QFT operator $\hat{F}$ is given by

$$
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
$$

#### One-Qubit System
For a one-qubit system, with $N = 2$ and $\omega_2 = e^{i\pi} = -1$, the QFT matrix reduces to

$$
  \mathbf{F}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix},
$$

which is precisely the Hadamard gate.

#### Two-Qubit System
For a two-qubit system, with $N = 4$ and $\omega_4 = e^{i \pi/2} = i$, the QFT matrix becomes

$$
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

As outlined in @book_scherer_2019 [pp. 241-245], the quantum Fourier transform can be realized using a sequence of Hadamard gates and controlled discrete phase shift gates. Applying the Hadamard gate to the computational basis states $\ket{0}$ and $\ket{1}$ produces the superpositions:

$$
  \hat{H}\ket{0} = \frac{\ket{0} + \ket{1}}{\sqrt{2}},\quad \hat{H}\ket{1} = \frac{\ket{0} - \ket{1}}{\sqrt{2}}.
$$

This can be generalized using the identity $e^{i\pi} = -1$, so that for $x_j \in\set{0,1}$:

$$
  \hat{H}\ket{x_j} = \frac{\ket{0} + e^{i\pi x_j} \ket{1}}{\sqrt{2}}
$$

In terms of the binary fraction $0_{.x_j} = x_j/2$, this becomes

$$
  \hat{H} = \frac{\ket{0} + e^{i 2\pi 0_{.x_j}} \ket{1}}{\sqrt{2}}.
$$

Now consider an $n$-qubit register in the state $\ket{x} = \ket{x_{n-1} \dots x_0}$. Applying a Hadamard gate to most significant qubit $\ket{x_{n-1}}$ yields

$$
  \hat{H}_{n-1} \ket{x} = \frac{\ket{0} + e^{i 2\pi 0_{.x_{n-1}}} \ket{1}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0}.
$$

To accumulate the desired relative phases for the QFT, we apply a sequence of controlled phase shift gates. A controlled phase gate $\hat{R}_{jk}$, where qubit $k$ is the control and qubit $j$ is the target, introduces a relative phase $e^{i\pi/2^{j-k}}$ to the $\ket{1}$ component of the target qubit if the control qubit is in state $\ket{1}$. Applying $\hat{R}_{n-1,n-2}$ to $\hat{H}_{n-1}\ket{x}$ gives

$$
\begin{align*}
  \hat{R}_{n-1,n-2} \hat{H}_{n-1} \ket{x} =& \frac{\ket{0} + \exp\left(i 2\pi 0_{.x_{n-1}} + i\pi \frac{x_{n-2}}{2}\right) \ket{1}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0} \\
  =& \frac{\ket{0} + e^{i 2\pi 0_{.x_{n-1} x_{n-2}}}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0},
\end{align*}
$$

Applying the remaining controlled phase gates to $\ket{x_{n-1}}$ results in

$$
\begin{align*}
  \hat{R}_{n-1,0} \cdots \hat{R}_{n-1,n-2} \hat{H}_{n-1} \ket{x} = \frac{\ket{0} + e^{i 2\pi 0_{.x_{n-1} \dots x_0}}}{\sqrt{2}} \otimes \ket{x_{n-2} \dots x_0}.
\end{align*}
$$

Repeating this process for each remaining factor qubit in $\ket{x_{n-2} \dots x_0}$ produces the QFT up to qubit order:

$$
  \prod_{j=0}^{n-1} \left(\left[\prod_{k=0}^{j-1} \hat{R}_{jk} \right] \hat{H}_j \right) = \frac{1}{\sqrt{2^n}} \bigotimes_{k=n-1}^0 \left[\ket{0} + e^{i2\pi 0_{.x_k \dots x_0}} \ket{1} \right].
$$

Finally, applying a series of swap gates to reverse the order of the qubits completes the implementation of the QFT. The circuit diagram in [](#figure:fourier-circuit) shows the construction described above. 

```{figure} figures/fourier_circuit.pdf
:label: figure:fourier-circuit
:align: center

Quantum circuit for the quantum Fourier transform, constructed using Hadamard gates, controlled discrete phase shift gates, and final swap gates to reverse qubit order.
```

# Appendix
