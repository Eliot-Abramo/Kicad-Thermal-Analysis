# Physics and numerical methods for PCB thermal analysis in vacuum

**The thermal behavior of PCBs in thermal vacuum is governed by anisotropic conduction through copper-FR4 laminates, T⁴ radiation exchange, and I²R Joule heating — a coupled nonlinear problem solvable with established finite-difference methods, Picard or Newton iteration for radiation, and PCG linear solvers.** Every formula, material constant, and numerical technique cited below traces to peer-reviewed sources, IPC standards, or NASA/ESA technical publications. This reference document covers the ten foundational topics needed for a rigorous physics documentation section, with specific equations, standard parameter values, convergence properties, and validation criteria drawn from Incropera, Gilmore's *Spacecraft Thermal Control Handbook*, ECSS handbooks, and IPC standards.

---

## 1. Governing heat equation and finite-difference discretization

The three-dimensional transient heat equation with spatially varying, anisotropic thermal conductivity and volumetric heat generation is:

$$\rho \, c_p \, \frac{\partial T}{\partial t} = \frac{\partial}{\partial x}\!\left(k_x \frac{\partial T}{\partial x}\right) + \frac{\partial}{\partial y}\!\left(k_y \frac{\partial T}{\partial y}\right) + \frac{\partial}{\partial z}\!\left(k_z \frac{\partial T}{\partial z}\right) + \dot{Q}$$

where ρ is density, c_p is specific heat, k_x, k_y, k_z are directional thermal conductivities, and Q̇ is volumetric heat generation (W/m³). This is derived from Fourier's law combined with conservation of energy. Standard derivations appear in **Incropera et al., *Fundamentals of Heat and Mass Transfer*, 8th ed. (Wiley, 2017), Chapter 2**, and **Çengel & Ghajar, *Heat and Mass Transfer*, 6th ed. (McGraw-Hill, 2020), Chapter 2**. The **NASA Passive Thermal Control Engineering Guidebook, v4.0 (NTRS 20230013900, September 2023)** references this equation in Section 4.2 and explicitly discusses PCB material property calculation (Figure 14, p. 50).

**Finite-difference discretization on non-uniform grids.** For a conservative (finite-volume) discretization with non-uniform node spacings h₋ = x_i − x_{i−1} and h₊ = x_{i+1} − x_i, the spatial operator becomes:

$$\frac{1}{\Delta x_i}\left[\frac{k_{i+1/2}(T_{i+1} - T_i)}{h_+} - \frac{k_{i-1/2}(T_i - T_{i-1})}{h_-}\right]$$

where Δx_i is the control volume width centered on node i, and k_{i±1/2} are interface conductivities. This reduces to the standard central-difference stencil (1/h²)[T_{i−1} − 2T_i + T_{i+1}] on uniform grids. The seminal reference is **Patankar, *Numerical Heat Transfer and Fluid Flow* (Hemisphere/McGraw-Hill, 1980)**, which introduced the conservative finite-volume discretization and the harmonic-mean interface treatment.

**Harmonic mean thermal conductivity at material interfaces.** When adjacent cells have different conductivities k_i and k_{i+1}, the effective interface conductivity preserving heat-flux continuity is the harmonic mean:

$$k_{i+1/2} = \frac{2 \, k_i \, k_{i+1}}{k_i + k_{i+1}}$$

The physical basis is that the two half-cells act as thermal resistances in series: heat must traverse both materials sequentially. **If one material is an insulator (k → 0), the harmonic mean correctly yields zero flux**, unlike the arithmetic mean which would give a nonphysical nonzero value. Patankar's 1978 IHTC paper ("A numerical method for conduction in composite materials") first demonstrated this property. A rigorous comparative study by **Kadioglu et al. (INL/EXT-08-13999, Idaho National Laboratory, 2008)** confirmed that harmonic averaging is preferred for multi-material problems because it preserves the zero-flux limit at insulator boundaries.

---

## 2. Effective thermal conductivity of copper-FR4 laminates

PCBs are **strongly anisotropic** due to their layered copper-FR4 structure, with in-plane conductivity exceeding through-plane conductivity by two to three orders of magnitude. The correct mixing rules depend on the direction of heat flow relative to the laminate layers.

**In-plane (x,y) — parallel mixing rule (Voigt upper bound).** Copper and FR4 provide parallel heat-flow paths within each layer. The effective conductivity is the volume-fraction-weighted sum:

$$k_{\text{eff,in-plane}} = f \cdot k_{\text{Cu}} + (1 - f) \cdot k_{\text{FR4}}$$

where f is the copper fill fraction. For f = 0.5 with k_Cu = 385 W/(m·K) and k_FR4 = 0.25 W/(m·K), this yields **k_eff,xy ≈ 192.6 W/(m·K)**.

**Through-plane (z) — series mixing rule (Reuss lower bound).** Heat must traverse each layer sequentially, so the layers act as thermal resistances in series:

$$\frac{1}{k_{\text{eff,through}}} = \frac{f}{k_{\text{Cu}}} + \frac{1 - f}{k_{\text{FR4}}}$$

For the same parameters, **k_eff,z ≈ 0.50 W/(m·K)** — a ratio of nearly 400:1 relative to the in-plane value. This extreme anisotropy is confirmed by industry analysis tools (Cadence, SimScale) and the NASA Passive Thermal Control Engineering Guidebook (Figure 14, p. 50).

These rules follow directly from composite theory. The parallel model is the Voigt upper bound; the series model is the Reuss lower bound. Standard derivations appear in **Incropera, Chapter 3** (composite wall thermal resistance networks) and in composite materials literature (Hashin & Shtrikman, *J. Appl. Phys.* 33(10), 1962). **Dede et al., "Design of Anisotropic Thermal Conductivity in Multilayer Printed Circuit Boards," *IEEE Trans. CPMT* (2015)** explicitly validates the mixing rules for PCB laminates.

**Important caveat:** FR4 itself is anisotropic due to its woven glass-fiber structure: in-plane k ≈ 0.6–0.8 W/(m·K), through-plane k ≈ 0.2–0.4 W/(m·K). The commonly cited **k_FR4 = 0.29 W/(m·K)** refers to the through-plane value and is a reasonable representative of the published 0.25–0.40 range.

---

## 3. Radiation heat transfer in thermal vacuum

In vacuum, convection is absent. Heat rejection occurs exclusively through radiation, governed by the Stefan-Boltzmann law. The **Stefan-Boltzmann constant σ = 5.670374419 × 10⁻⁸ W/(m²·K⁴)**.

**General two-surface enclosure formula.** For a gray, diffuse, opaque surface (1) enclosed by surface (2), the Oppenheim radiation network method gives:

$$Q = \frac{\sigma(T_1^4 - T_2^4)}{\frac{1-\varepsilon_1}{\varepsilon_1 A_1} + \frac{1}{A_1 F_{12}} + \frac{1-\varepsilon_2}{\varepsilon_2 A_2}}$$

**Small-body-in-large-enclosure simplification.** For a PCB inside a TVAC chamber, two conditions are met. First, a flat plate (or any convex body) has self-view-factor F₁₁ = 0, so by the summation rule **F₁₂ = 1 exactly** — this is not an approximation. Second, when A₁ ≪ A₂ (a ~100 cm² PCB in a ~1 m² shroud gives A₁/A₂ ≈ 0.01), the enclosure surface resistance term (1−ε₂)/(ε₂A₂) → 0. The formula reduces to:

$$\boxed{Q = \varepsilon_1 \cdot \sigma \cdot A_1 \cdot (T_1^4 - T_2^4)}$$

**The enclosure emissivity drops out entirely** — only the PCB surface emissivity matters. The physical interpretation: the large enclosure acts as an effective blackbody because negligible reflected energy returns to the small body. This derivation is standard in **Incropera, Chapter 13** and **Çengel, Chapter 15**. The **Spacecraft Thermal Control Handbook (Gilmore, 2nd ed., 2002)** uses this formulation throughout.

**Standard emissivity values for space thermal engineering**, drawn from the Design1st spacecraft thermal database, UPM thermo-optical property tables, and the Spacecraft Thermal Control Handbook (Gilmore, Ch. 4), are:

| Surface | Total hemispherical emissivity ε | Notes |
|---------|----------------------------------|-------|
| Solder mask (green/black) | **0.85–0.92** | Organic polymer; high IR emissivity regardless of visible color |
| Bare polished copper | **0.02–0.05** | Highly reflective in IR |
| Oxidized copper (thick oxide) | **0.50–0.78** | Depends on oxide thickness |
| FR4 (glass-reinforced polymer) | **0.85–0.90** | Similar to other organic/GFRP composites |
| Anodized aluminum (clear) | **0.76–0.87** | Standard spacecraft internal finish |
| Black anodized aluminum | **0.82–0.88** | Common for spacecraft structural panels |
| Bare Kapton polyimide | **0.90–0.95** | Bare film without metallic backing |

The value **solder mask ε = 0.90** used in typical analyses is accurate and falls squarely within the published range. **ECSS-Q-ST-70-09C** defines the ESA standard measurement procedure for thermo-optical properties.

---

## 4. Picard iteration and Newton linearization for the T⁴ nonlinearity

The T⁴ radiation term is one of the strongest nonlinearities in thermal analysis. Two standard approaches exist for solving the resulting nonlinear system.

**(a) Picard iteration (successive substitution / lagged T⁴).** At iteration k+1, evaluate the radiation term using temperatures from iteration k, converting the problem to a linear system:

$$\mathbf{K} \cdot \mathbf{T}^{k+1} + \varepsilon \sigma A \, (T^k)^4 = \mathbf{f} + \varepsilon \sigma A \, T_\infty^4$$

Alternatively, use the algebraic identity a⁴ − b⁴ = (a−b)(a+b)(a²+b²) to define a pseudo-convection coefficient h_rad = εσF(T_k² + T_∞²)(T_k + T_∞), converting the radiation BC into a Robin-type condition that is linear in T. Picard iteration is **linearly convergent** (first-order), has a **large radius of convergence**, and preserves the symmetry of the system matrix.

**(b) Newton linearization.** Expand T⁴ in a first-order Taylor series about the current iterate:

$$T^4 \approx (T^k)^4 + 4(T^k)^3(T - T^k) = 4(T^k)^3 T - 3(T^k)^4$$

This adds **4εσA(T^k)³ to the conductance matrix diagonal** (the tangent stiffness) and adds **3εσA(T^k)⁴ to the right-hand side**. Newton's method is **quadratically convergent** near the solution but has a smaller convergence basin and requires Jacobian assembly at each iteration.

**(c) Convergence comparison.** Newton converges in fewer iterations when the initial guess is close to the solution, but can diverge from poor starting points. The standard practical strategy, documented by **Lu et al. (*Nuclear Engineering and Design*, 2023)** and **Langtangen's PDE course notes**, is a **combined Picard-Newton approach**: start with a few Picard iterations to reach the convergence basin, then switch to Newton for quadratic convergence.

**Under-relaxation** is essential for stability: T_new = ω·T_computed + (1−ω)·T_old, with **ω = 0.5–0.8** typical for radiation problems. Convergence is monitored via the relative temperature change ||T^{k+1} − T^k||/||T^k|| < tol, with tol = 10⁻⁴ to 10⁻⁶ standard.

Key references include **Stelzer (1987), "Experiences in non-linear analysis of temperature fields with finite elements," *Int. J. Numer. Methods Eng.* 24(1)**; **Hughes & Winget (1985), *Comput. Methods Appl. Mech. Eng.* 48(1)**, which combines modified Newton-Raphson with element-by-element PCG; and the **Sandia Aria thermal code documentation (v5.24, Section 3.6)**, which details the Newton Jacobian contribution from radiative flux.

---

## 5. Thermal via barrel conductance

The thermal resistance of a plated through-hole (PTH) via follows directly from Fourier's law:

$$R_{\text{via}} = \frac{L}{k_{\text{Cu}} \cdot A_{\text{cross}}}$$

where L is the board thickness and A_cross is the copper barrel cross-sectional area.

**The exact cross-sectional area** of the annular copper barrel is:

$$A = \frac{\pi}{4}\left[d_{\text{drill}}^2 - (d_{\text{drill}} - 2t)^2\right] = \pi \, t \, (d_{\text{drill}} - t)$$

where d_drill is the drill diameter (= barrel outer diameter) and t is the plating thickness. The approximate form **A ≈ π·d_drill·t** is valid when t ≪ d_drill, introducing ~5–12% error for typical geometries (t = 25 µm, d_drill = 200–400 µm). CircuitCalculator.com and Fineline Global both confirm the exact formula in the equivalent form A = π·b·(d+b), where d is the finished inner diameter and b = t is the plating thickness.

**Standard plating thickness per IPC-6012** (Qualification and Performance Specification for Rigid PCBs):

- **IPC Class 2: ≥ 20 µm (0.8 mil)** average copper plating in PTH
- **IPC Class 3: ≥ 25 µm (1.0 mil)** average copper plating in PTH
- Space applications typically require **IPC-6012 with /S (space) or /DS (defense/space) addendum**, mandating Class 3 plating at minimum

**Worked example** (from Fineline Global): For b = 25 µm plating, d = 250 µm finished hole, L = 1.6 mm board thickness, k_Cu = 385 W/(m·K): A = π × 275 × 25 = 21,598 µm², yielding **R_via ≈ 193 K/W per via**. Texas Instruments corroborates this order of magnitude (~100–125 °C/W per via depending on copper weight) in application note SLPA015.

For **N parallel thermal vias**, R_array = R_single/N. The air column inside an unfilled via (k_air ≈ 0.026 W/(m·K)) contributes negligibly — approximately six orders of magnitude higher resistance than the copper barrel. **Copper-filled vias** conduct through the full cross-section (A = π/4·d_drill²), reducing thermal resistance dramatically.

---

## 6. I²R Joule heating in PCB traces

**Trace resistance** follows the standard formula for a rectangular conductor:

$$R = \frac{\rho \, L}{w \, t}$$

where ρ is copper resistivity, L is trace length, w is trace width, and t is trace thickness. The standard conversion is **1 oz/ft² copper = 35 µm (1.37 mil)** thickness.

**Copper resistivity and temperature coefficient.** Per **IEC 60028** and the IACS (International Annealed Copper Standard), pure annealed copper at 20°C has:

- **ρ₀ = 1.724 × 10⁻⁸ Ω·m**
- **α = 0.00393/°C** (temperature coefficient at 20°C reference)

The temperature-dependent form is **ρ(T) = ρ₀·[1 + α·(T − 20°C)]**. The sometimes-cited value α = 0.00385/°C applies to slightly less pure commercial copper (~97.3% IACS conductivity). For PCB analysis, **0.00393/°C is the correct IACS value**, confirmed by NBS Handbook-100 and Fisk Alloy Wire technical documentation. Electroplated copper in via barrels has higher resistivity (~1.9 × 10⁻⁸ Ω·m) due to grain structure.

**Power dissipation** is P = I²R = I²ρL/(wt). The **volumetric heat generation rate** for FEA modeling is:

$$\dot{Q}''' = \frac{I^2 \rho}{(w \cdot t)^2} \quad [\text{W/m}^3]$$

This is independent of trace length and represents a uniform volumetric source within the trace cross-section. The temperature-dependent resistivity creates positive feedback: higher temperature → higher resistance → more heating → requiring iterative coupling between electrical and thermal solutions.

**IPC-2152 (2009)** supersedes the legacy IPC-2221 charts for trace current capacity. The older IPC-2221 formula I = K·ΔT⁰·⁴⁴·A_c⁰·⁷²⁵ (with K = 0.048 for external layers, K = 0.024 for internal layers) was based on 50-year-old single-board data. IPC-2152 provides empirically derived charts (Figure 5-2) with correction factors for board material, copper plane proximity, board thickness, and copper weight. Key finding: **internal traces carry current much closer to external traces** than the old 50% derating suggested.

---

## 7. Preconditioned Conjugate Gradient solver

The thermal conductance matrix from FDM discretization is **symmetric positive definite (SPD)**: symmetric from the reciprocity of Fourier's law, and positive definite because ρc_p > 0 and the diffusion operator is non-negative. The combined system matrix [C/Δt + θK] for implicit time integration inherits SPD character. This makes the Conjugate Gradient (CG) method the natural iterative solver.

The convergence bound for PCG is:

$$\frac{\|e_k\|_A}{\|e_0\|_A} \leq 2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$$

where κ = κ(M⁻¹A) = λ_max/λ_min is the condition number of the preconditioned system. **Effective preconditioning reduces κ**, accelerating convergence from O(n) iterations (unpreconditioned) toward O(√n) or better.

**Standard preconditioners ranked by effectiveness for thermal problems:**

- **Jacobi (diagonal):** M = diag(A). Trivially parallelizable, minimal storage. Effective when the matrix is diagonally dominant (common with small time steps). Modest condition number reduction.
- **SSOR:** Better conditioning improvement; inherently sequential.
- **Incomplete Cholesky IC(0):** Computes an approximate L·Lᵀ factorization retaining only the original sparsity pattern. **Generally the most effective for SPD thermal systems** — significant κ reduction at moderate cost. Higher-order variants IC(k) allow fill-in for further improvement.

**Standard convergence criterion** is ||r_k||/||r₀|| < tol, where **tol = 10⁻⁶ to 10⁻¹⁰** depending on required accuracy. Each iteration costs O(nnz) operations for the matrix-vector product.

Key references: **Shewchuk, "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" (CMU-CS-94-125, 1994)**; **Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (SIAM, 2003)**; **Golub & Van Loan, *Matrix Computations*, 4th ed. (JHU Press, 2013)**. The **Hughes & Winget (1985)** paper in *CMAME* demonstrated element-by-element PCG specifically for nonlinear transient thermal problems with radiation BCs.

---

## 8. Crank-Nicolson time integration and the θ-method

The Crank-Nicolson scheme applies the trapezoidal rule to the semi-discrete thermal system C·dT/dt = K·T + f:

$$\left[\frac{\mathbf{C}}{\Delta t} + \frac{\mathbf{K}}{2}\right]\mathbf{T}^{n+1} = \left[\frac{\mathbf{C}}{\Delta t} - \frac{\mathbf{K}}{2}\right]\mathbf{T}^n + \frac{1}{2}(\mathbf{f}^{n+1} + \mathbf{f}^n)$$

This is a special case (θ = 0.5) of the **generalized θ-method**, where θ = 0 gives forward Euler, θ = 1 gives backward Euler. Key properties of the Crank-Nicolson scheme:

- **Second-order temporal accuracy**: O(Δt²), vs O(Δt) for forward/backward Euler
- **Unconditional stability**: the θ-method is A-stable for all θ ≥ 0.5, so there is no CFL-type stability restriction on Δt
- **Known oscillation issue**: when the mesh Fourier number Fo = αΔt/Δx² exceeds ~0.5, the amplification factor G = (1 − 2Fo)/(1 + 2Fo) becomes negative, causing **decaying spurious oscillations** near discontinuities or sharp gradients. This is not an instability — solutions remain bounded — but produces non-physical temperature undershoots/overshoots.

**Remedies for oscillations:** Using **θ > 0.5** (e.g., θ = 0.55 or the Galerkin value θ = 2/3) damps oscillations at the cost of reduced temporal accuracy. Backward Euler (θ = 1) eliminates oscillations entirely but is only first-order. Adaptive time stepping — keeping Fo ≤ 0.5 during rapid transients, then increasing Δt when the solution smooths — provides the practical best of both worlds.

The original paper is **Crank & Nicolson, *Proc. Cambridge Phil. Soc.* 43(1), 50–67, 1947**. Modern treatments appear in **Incropera, Chapter 5** (transient conduction with FD methods) and the **NTNU Numerical Methods for Engineers** lecture series (Section 7.5.3).

---

## 9. Energy balance verification and model validation

**Energy balance at steady state** requires ΣQ_in = ΣQ_out. The energy balance residual is:

$$\text{Residual} = \frac{|Q_{\text{in}} - Q_{\text{out}}|}{\max(Q_{\text{in}}, Q_{\text{out}})} \times 100\%$$

For transient analysis, Q_in − Q_out = dE_stored/dt, where E_stored = Σ(ρ·c_p·V·T). The **ECSS-E-HB-31-03A Thermal Analysis Handbook (2016), Section 5.3** states that in a well-converged model, key outputs should be independent of further tightening of convergence criteria, and notes that "hard numerical guidelines cannot easily be established" because criteria are model-dependent. In practice:

- **< 1% residual** is the accepted threshold for engineering use
- **< 0.1% residual** is expected for high-fidelity or cryogenic applications

Additional verification methods include comparison with analytical solutions (1D bar, lumped capacitance, infinite fin), grid convergence studies (**ECSS-E-HB-31-03A Section 4.5.2** on mesh independence), time-step convergence checks (Section 4.5.4), and view-factor reciprocity/closure verification (Section 5.5). The ECSS handbook explicitly references the **ASME V&V Guide** as the formal verification and validation framework.

---

## 10. Industrial validation standards for space flight hardware

Three tiers of standards govern thermal model validation for space electronics, with progressively specific requirements.

**ECSS-E-ST-31C "Thermal Control" (2008)** is the primary ESA standard. It mandates thermal balance testing (TBT) for all flight hardware controlled by radiative/conductive exchange, requiring at least two steady-state cases (hot and cold worst-case) plus a transient case for dynamically sensitive items. Correlation criteria per ECSS practice are: **average deviation ≤ 2°C, standard deviation ≤ 3°C, individual sensor maximum < 5°C**.

**SMC-S-016 (2014) / MIL-STD-1540E** imposes the most stringent correlation criterion: **all temperature predictions within ±3°C of test data**. The rationale: with a ±11°C thermal uncertainty margin (providing 95% confidence), a ±3°C correlation error consumes only 27% of the margin. The landmark **Welch (Aerospace Corporation) ICES-2016-006 paper** compares correlation criteria across organizations and shows that relaxing to ±5°C drops the probability of staying within temperature limits from 93% to 64%.

**GSFC-STD-7000B (GEVS)** defines thermal margins for flight predictions: **≥10°C qualification margin** beyond predicted temperatures for passively controlled components, **≥5°C for actively controlled components**. Uncorrelated models require **±20°C qualification margin**. Temperature stabilization criteria vary: MIL-STD-1540E requires < 0.2°C/hr over 5 hours; NASA Goddard uses < 0.05°C/hr over ≥ 6 hours.

**Tool validation responsibility** is clarified in ECSS-E-HB-31-03A: **code verification is the software vendor's responsibility**; users are responsible for calculation verification (ensuring their model correctly captures the intended physics). The standard space thermal tools — **SINDA/FLUINT** (NASA heritage, CRTech; validated in Keller & Vogel, SAE 961452, 1996) and **ESATAN-TMS** (ESA standard, ITP Engines UK) — have extensive flight heritage and published validation against analytical benchmarks.

**NASA-STD-5009 is not relevant** to thermal analysis — it covers nondestructive evaluation for fracture-critical metallic components. The correct NASA thermal references are NASA-STD-7002B (Payload Test Requirements) and the NASA Passive Thermal Control Engineering Guidebook (NTRS, Rev 4.0/5.1, 2023).

---

## Material property values confirmed against published literature

The table below summarizes verified thermal properties at approximately 300K (room temperature), with assessment of commonly used values:

| Material | Property | Commonly used | Published reference value | Assessment |
|----------|----------|---------------|--------------------------|------------|
| FR4 (through-plane) | k | 0.29 W/(m·K) | 0.25–0.40 W/(m·K) | **Accurate** — mid-range |
| FR4 (in-plane) | k | — | 0.6–1.0 W/(m·K) | FR4 itself is anisotropic |
| Pure copper | k | 385 W/(m·K) | **401 W/(m·K)** at 300K | Slightly low; 401 is the Incropera value for OFHC copper |
| Electroplated Cu (PCB) | k | 385 W/(m·K) | 370–392 W/(m·K) | Reasonable for PCB copper |
| Aluminum 6061-T6 | k | 167 W/(m·K) | 152–167 W/(m·K) | Range reflects alloy variability |
| Solder mask | ε (IR) | 0.90 | 0.85–0.92 | **Accurate** |
| Solder mask | k | — | 0.20–0.25 W/(m·K) | Thermally thin (~20 µm); minimal conductive impact |

**Temperature dependence over −40°C to +125°C is modest for all key materials.** Copper k varies ~5% (413 → 393 W/(m·K) from 233K to 398K). Aluminum 6061-T6 k varies ~10% (148 → 163 W/(m·K)). FR4 k has weak temperature dependence with insufficient published data for precision modeling; constant-value treatment is standard practice. Specific heat generally increases with temperature per the Debye model: copper c_p ranges from ~360 to ~398 J/(kg·K) over this interval.

Standard material property references include **Incropera, Appendix A tables**; **NIST Cryogenic Properties Database** (Al 6061-T6 polynomial fits); **MMPDS (formerly MIL-HDBK-5)** for metallic alloys; and the **Spacecraft Thermal Control Handbook (Gilmore, 2002), Appendix B**. For space-specific surface properties, **ECSS-Q-ST-70-09C** defines the measurement standard, and the UPM thermo-optical properties database provides beginning-of-life and end-of-life values for spacecraft surface finishes.

---

## Conclusion

The physics of PCB thermal analysis in TVAC conditions rests on a well-established foundation. The **anisotropic effective conductivity** of copper-FR4 laminates follows from elementary composite theory (parallel for in-plane, series for through-plane), with the harmonic mean correctly handling material interfaces in finite-difference discretization (Patankar, 1980). The **small-body-in-large-enclosure radiation formula** Q = εσA(T⁴ − T_enc⁴) is exact for convex bodies when A₁/A₂ ≪ 1 — a condition easily met in TVAC testing. For numerical solution, the **combined Picard-Newton strategy** provides robust convergence of the radiation nonlinearity, PCG with Incomplete Cholesky preconditioning efficiently solves the resulting SPD systems, and Crank-Nicolson provides second-order transient accuracy if oscillation precautions are taken (θ ≥ 0.55 or adaptive time stepping). The commonly assumed material values (FR4 k = 0.29, solder mask ε = 0.90) are confirmed by published literature, though **pure copper k = 401 W/(m·K)** is more precise than the widely used 385. Validation against ECSS and MIL-STD criteria (±3°C correlation, ±10°C qualification margin) provides the framework for qualifying thermal models for flight hardware.