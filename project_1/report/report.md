---
title: FYS5419 - Project 1
authors:
  - name: Insert Name
site:
  template: article-theme
exports:
  - format: pdf
    template: ../report_template
    output: report.pdf
    showtoc: true
math:
  # Note the 'single quotes'
  '\argmin': '\operatorname{argmin}'
  '\R': '\mathbb{R}'
  '\Set': '{\left\{ #1 \right\}}'
bibliography: references.bib
abstract: |
---

# Part a)

## One-qubit operations

I have implemented a Python class representing a single qubit $\ket{\psi}$ as a 2D complex vector

$$
  \ket{\psi} = \cos\left(\frac{\theta}{2})\ket{0} + e^{i\phi} \sin\left(\frac{\theta}{2}\right) \ket{1}
$$

where

$$
  \ket{0} = \begin{bmatrix} 1 // 0 \end{bmatrix}, \quad \ket{1} = \begin{bmatrix} 0 // 1 \end{bmatrix}
$$


## Bell states

We let $H = H_0 + H_I$, where

$$
  H_0 = \begin{bmatrix} E_1 & 0 \\ 0 & E_2 \end{bmatrix}
$$

and

$$
  H_I = \begin{bmatrix} V_{11} & V_{12} \\ V_{21} & V_{22} \end{bmatrix}
$$

We rewrite $H$ (and $H_0$ and $H_I$) via Pauli matrices

$$
  H_0 = \mathcal{E} I + \Omega \sigma_z, \quad \mathcal{E} = \frac{E_1\n + E_2}{2}, \; \Omega = \frac{E_1-E_2}{2}
$$

and for the interaction part

$$
  H_I = c \boldsymbol{I}_2 + \omega_z \boldsymbol{\sigma}_z + \omega_x \boldsymbol{\sigma}_x
$$

with $c = (V_{11}+V_{22})/2$, $\omega_z = (V_{11}-V_{22})/2$ and $\omega_x = V_{12}=V_{21}$. We let our Hamiltonian depend linearly on a strength parameter $\lambda$

$$
  H = H_0 + \lambda H_\mathrm{I}
$$

# Part b)

In terms of the Pauli matrices, the total Hamiltonian $\hat{H} = \hat{H}_0 + \hat{H}_\text{I}$ can be written

$$
  \hat{H} = (\mathcal{E} + \lambda c)\hat{I}_2 + (\Omega + \lambda\omega_z) \sigma_z + \lambda\omega_x \sigma_x
$$

The identity matrix $\hat{I}$ contributes only a constant shift $\mathcal{E} + \lambda c$ to the eigenvalues, so the relevant part of the Hamiltonian for computing eigenvalues is

$$
  \hat{H}' = (\Omega + \lambda\omega_z) \sigma_z + \lambda\omega_x \sigma_x,
$$

or in matrix form

$$
  \hat{H}' = \begin{bmatrix} \Omega + \lambda\omega_z & \lambda \omega_x \\ \lambda\omega_x & -(\Omega + \lambda\omega_z) \end{bmatrix}
$$

To find the eigenvalues, we solve the characteristic equation $\det(\hat{H}' - E'\hat{I}) = 0$:

$$
\begin{align*}
  0 =& \begin{vmatrix} \Omega + \lambda\omega_z - E' & \lambda\omega_x \\ \lambda\omega_x & -(\Omega + \lambda\omega_z) - E' \end{vmatrix} \\
  =& (\Omega + \lambda\omega_z - E')(-(\Omega + \lambda\omega_z) - E') - (\lambda\omega_x)2 \\
  =& -(\Omega + \lambda\omega_z - E')(\Omega + \lambda\omega_z + E') - (\lambda\omega_x)2.
\end{align*}
$$

Using the identity $(a - b)(a + b) = a^2 - b^2$, we get

$$
  -(\Omega + \lambda\omega_z)^2 + E'^2 - (\lambda\omega_x)^2 = 0
$$

or, after rearranging

$$
  E'^2 = (\Omega + \lambda\omega_z)^2 + (\lambda\omega_x)^2.
$$

Taking the square root, we find the reduced eigenvalues

$$
  E'_\pm = \pm \sqrt{(\Omega + \lambda \omega_z)^2 + (\lambda\omega_x)^2}
$$

Adding the constant shift $\mathcal{E} + \lambda c$, we finally arrive at

$$
  E_\pm = \mathcal{E} + \lambda c \pm \sqrt{(\Omega + \lambda \omega_z)^2 + (\lambda\omega_x)^2}
$$



We find the eigenvalues of $\hat{H}$ by solving the characteristic equation $\det(\hat{H} - E\hat{I}_2) = 0$:

$$
\begin{align*}
  0 =& \begin{vmatrix} E_1 + \lambda V_{11} - E & \lambda V_{12} \\ \lambda V_{21} & E_2 + \lambda V_22 - E \end{vmatrix} \\
  =& (E_1 - \lambda V_{11} - E)(E_2 + \lambda V_{22} - E) - \lambda^2 V_{12} V_{21} \\
  =& E^2 - E[E_1 + E_2 + \lambda (V_{11} + V_{22})] + (E_1 + \lambda V_{11})(E_2 + \lambda V_{22}) - \lambda^2 V_{12}V_{21}
\end{align*}
$$

Solving for $E$ using the quadratic formula yields

$$
\begin{align*}
  E =& \frac{E_1 + E_2 + \lambda (V_{11} + V_{22}) \pm\sqrt{[E_1 + E_2 + \lambda (V_{11} + V_{22})] - 4 \lambda^2 V_{12} V_{21}}}{2} \\
  =& \mathcal{E} + \lambda c \pm \sqrt{\mathcal{E}^2 + \lambda \mathcal{E}c + \lambd^2 c^2 - \lambda^2 \omega_x^2},
\end{align*}
$$

where we have used the substitutions

$$
  \mathcal{E} = \frac{E_1 + E_2}{2},\quad c = \frac{V_{11} + V_{22}}{2}, \quad \omega_x = V_{12} = V_{21}
$$

Setting the parameters $E_1 = 0$, $E_2 = 4$, $V_{11} = −V_{22} = 3$ and $V_12 = V_21 = 0.2$

# Appendix

## Code Repository

The Python source code used for this project is available at [https://github.com/semapheur/fys5419](https://github.com/semapheur/fys5419).