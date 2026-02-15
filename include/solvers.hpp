/**
 * @file solvers.hpp
 * @brief Solveurs linéaires et accélérateurs pour systèmes mixtes RTₖ-Pₘ
 * 
 * Ce module implémente les stratégies de résolution pour le système selle issu
 * de la discrétisation mixte-duale de l'équation de diffusion neutronique.
 * 
 * =============================================================================
 * CONTEXTE PHYSIQUE
 * =============================================================================
 * 
 * L'équation de diffusion neutronique multigroupe :
 * 
 *     -∇·(Dᵍ∇φᵍ) + Σᵣᵍφᵍ = Σₕχᵍ(νΣf)ʰφʰ/k + Σₕ Σₛᵍ←ʰ φʰ + Qᵍ
 * 
 * où :
 *   - Dᵍ     : coefficient de diffusion du groupe g [cm]
 *   - φᵍ     : flux scalaire du groupe g [n/cm²/s]
 *   - Σᵣᵍ    : section efficace de retrait [cm⁻¹]
 *   - χᵍ     : spectre de fission (probabilité d'émission dans le groupe g)
 *   - νΣf    : section de production [cm⁻¹]
 *   - k      : facteur de multiplication effectif (valeur propre)
 *   - Σₛᵍ←ʰ : section de transfert h→g [cm⁻¹]
 *   - Qᵍ     : source externe [n/cm³/s]
 * 
 * =============================================================================
 * FORMULATION MIXTE-DUALE (HÉBERT)
 * =============================================================================
 * 
 * La formulation variationnelle mixte introduit le courant J = -D∇φ :
 * 
 *     J + D∇φ = 0        (loi de Fick constitutive)
 *     ∇·J + Σᵣφ = S      (bilan neutronique)
 * 
 * La discrétisation RTₖ-Pₘ conduit au système selle :
 * 
 *     ┌       ┐ ┌   ┐   ┌   ┐
 *     │ A  Bᵀ │ │ J │   │ 0 │
 *     │ B  C  │ │ φ │ = │ f │
 *     └       ┘ └   ┘   └   ┘
 * 
 * avec les matrices élémentaires (intégrales sur l'élément K) :
 * 
 *     A[i,j] = ∫_K (1/D) ψᵢ · ψⱼ dV      [Matrice de masse RT, SPD]
 *     B[i,j] = ∫_K φⱼ ∇·ψᵢ dV            [Opérateur divergence]
 *     C[i,j] = Σᵣ ∫_K φᵢ φⱼ dV           [Matrice de réaction, SPD si Σᵣ>0]
 *     f[i]   = ∫_K S φᵢ dV               [Second membre]
 * 
 * =============================================================================
 * RÉSOLUTION PAR COMPLÉMENT DE SCHUR
 * =============================================================================
 * 
 * L'élimination de J donne le système réduit sur φ :
 * 
 *     S·φ = f   avec   S = C + B·A⁻¹·Bᵀ
 * 
 * Le complément de Schur S est :
 *   - Symétrique définie positive (si Σᵣ ≥ 0)
 *   - De dimension n_φ × n_φ (généralement << n_J)
 *   - Bien conditionnée pour les problèmes de diffusion
 * 
 * Deux stratégies de résolution :
 * 
 * 1) SCHUR EXPLICITE (solveurs directs, petits systèmes) :
 *    - Former S = C + B·A⁻¹·Bᵀ explicitement
 *    - Factoriser S (LU, LDLT, ou LLT)
 *    - Coût : O(n_φ · n_J²) pour former, O(n_φ³) pour factoriser
 * 
 * 2) SCHUR IMPLICITE (solveurs itératifs, grands systèmes) :
 *    - Ne JAMAIS former S explicitement
 *    - Produit S·x = C·x + B·(A⁻¹·(Bᵀ·x))
 *    - Coût par itération : O(nnz(A) + nnz(B) + nnz(C))
 *    - La factorisation LU de A est calculée une seule fois
 * 
 * Après résolution de φ, reconstruction de J :
 *     J = -A⁻¹·Bᵀ·φ
 * 
 * =============================================================================
 * ACCÉLÉRATION DES ITÉRATIONS EXTERNES
 * =============================================================================
 * 
 * Pour le problème k-effectif, on utilise la méthode des puissances :
 * 
 *     φ⁽ⁿ⁺¹⁾ = (1/k⁽ⁿ⁾) · L⁻¹ · F · φ⁽ⁿ⁾
 * 
 * où L = -∇·D∇ + Σᵣ (opérateur de fuite) et F = χ(νΣf)ᵀ (opérateur de fission).
 * 
 * Cette itération converge lentement (ratio de dominance ρ ≈ 1 pour réacteurs).
 * Deux accélérateurs sont proposés :
 * 
 * CHEBYSHEV : Accélération polynomiale
 *   - Exploite la structure du spectre de l'opérateur d'itération
 *   - Coefficients optimaux basés sur les polynômes de Tchebychev
 *   - Efficace quand le ratio spectral est connu
 * 
 * ANDERSON : Accélération de type quasi-Newton
 *   - Extrapolation basée sur l'historique des itérés
 *   - Robuste sans connaissance a priori du spectre
 *   - Paramètre m = profondeur de mémoire
 * 
 * =============================================================================
 * RÉFÉRENCES
 * =============================================================================
 * 
 * [1] Hébert A. (1993) "Application of a dual variational formulation to finite
 *     element reactor calculations", Ann. Nucl. Energy 20(12):823-845
 * [2] Hébert A. (2008) "A Raviart-Thomas-Schneider solution of the diffusion
 *     equation in hexagonal geometry", Ann. Nucl. Energy 35(3):363-376
 * [3] Walker H.F., Ni P. (2011) "Anderson acceleration for fixed-point
 *     iterations", SIAM J. Numer. Anal. 49(4):1715-1735
 * [4] Adams M.L., Morel J.E. (1993) "A two-grid acceleration scheme for the
 *     multigroup Sn equations with neutron upscattering"
 * 
 */

#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <vector>
#include <deque>
#include <memory>

// ============================================================================
// ALIAS DE TYPES EIGEN
// ============================================================================

/** @brief Vecteur dense double précision */
using Vec = Eigen::VectorXd;
using Vec_t = Eigen::VectorXd;

/** @brief Matrice dense double précision */
using Mat = Eigen::MatrixXd;

/** @brief Matrice creuse colonne-major (format CSC, optimal pour Eigen) */
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;

/** @brief Matrice creuse ligne-major (format CSR, optimal pour produits Ax) */
using SpMatR_t = Eigen::SparseMatrix<double, Eigen::RowMajor>;

/** @brief Triplet (i, j, value) pour assemblage de matrices creuses */
using Triplet = Eigen::Triplet<double>;

// ============================================================================
// ÉNUMÉRATIONS
// ============================================================================

/**
 * @enum LinearSolverType
 * @brief Types de solveurs linéaires disponibles pour le complément de Schur
 * 
 * Classification par famille :
 * 
 * SOLVEURS DIRECTS (factorisation complète) :
 *   - DIRECT_LU   : LU avec pivotage, usage général, robuste
 *   - DIRECT_LDLT : LDLᵀ pour matrices symétriques, 2× plus rapide que LU
 *   - DIRECT_LLT  : Cholesky pour matrices SPD, le plus rapide mais limité
 * 
 * SOLVEURS ITÉRATIFS (convergence en O(√κ) ou O(κ)) :
 *   - CG          : Gradient conjugué, optimal pour SPD
 *   - CG_DIAG     : CG + préconditionneur diagonal (Jacobi)
 *   - CG_ICHOL    : CG + Cholesky incomplet, très efficace pour diffusion
 *   - BICGSTAB    : Bi-CGSTAB, pour matrices non-symétriques
 *   - BICGSTAB_ILU: Bi-CGSTAB + ILU, robuste pour matrices mal conditionnées
 *   - LCG         : Moindres carrés conjugués, pour systèmes sur/sous-déterminés
 * 
 * Recommandations selon la taille du problème :
 *   - n < 1000      : DIRECT_LU (simplicité)
 *   - 1000 < n < 50k: DIRECT_LDLT (S est symétrique)
 *   - n > 50k       : CG_ICHOL ou BICGSTAB_ILU (mémoire)
 */
enum class LinearSolverType {
    // Solveurs directs (factorisation)
    DIRECT_LU,      ///< Factorisation LU avec pivotage partiel
    DIRECT_LDLT,    ///< Factorisation LDLᵀ (symétrique indéfinie)
    DIRECT_LLT,     ///< Factorisation Cholesky LLᵀ (SPD uniquement)
    
    // Solveurs itératifs (Krylov)
    CG,             ///< Gradient Conjugué sans préconditionnement
    CG_DIAG,        ///< CG + préconditionneur diagonal
    CG_ICHOL,       ///< CG + Cholesky incomplet
    BICGSTAB,       ///< Bi-CGSTAB sans préconditionnement
    BICGSTAB_DIAG,  ///< Bi-CGSTAB + préconditionneur diagonal
    BICGSTAB_ILU,   ///< Bi-CGSTAB + factorisation ILU incomplète
    LCG             ///< Moindres carrés conjugués (LSCG)
};

// ============================================================================
// SOLVEUR PAR COMPLÉMENT DE SCHUR
// ============================================================================

/**
 * @class SchurSolver
 * @brief Solveur par complément de Schur pour systèmes selle mixtes
 * 
 * Cette classe résout le système selle issu de la formulation mixte RTₖ-Pₘ :
 * 
 *     ┌       ┐ ┌   ┐   ┌   ┐
 *     │ A  Bᵀ │ │ J │   │ 0 │
 *     │ B  C  │ │ φ │ = │ f │
 *     └       ┘ └   ┘   └   ┘
 * 
 * par élimination de J via le complément de Schur S = C + B·A⁻¹·Bᵀ.
 * 
 * @section schur_algo Algorithme de résolution
 * 
 * 1. Factoriser A une fois (LU pour usage général)
 * 2. Résoudre S·φ = f par méthode directe ou itérative
 * 3. Reconstruire J = -A⁻¹·Bᵀ·φ
 * 
 * @section schur_implicit Produit Schur implicite
 * 
 * Pour les solveurs itératifs avec grands systèmes, le produit S·x est
 * calculé SANS former S explicitement :
 * 
 *     y = S·x = C·x + B·(A⁻¹·(Bᵀ·x))
 * 
 * Coût : une résolution LU (back-substitution) + produits matrice-vecteur.
 * 
 * @section schur_perf Complexité
 * 
 * Soit n_J = dim(J), n_φ = dim(φ), nnz = nombre de non-zéros.
 * 
 * | Opération           | Schur explicite      | Schur implicite     |
 * |---------------------|----------------------|---------------------|
 * | Formation de S      | O(n_φ · n_J²)        | 0                   |
 * | Factorisation       | O(n_φ³) ou O(nnz²)   | O(nnz(A))           |
 * | Résolution          | O(n_φ²)              | O(k · nnz) par iter |
 * | Mémoire             | O(n_φ²)              | O(nnz)              |
 * 
 * @note Pour les problèmes de diffusion typiques, n_φ ≈ n_éléments × (m+1)^d
 *       et n_J ≈ d × n_faces × (k+1)^(d-1), donc n_J >> n_φ en général.
 * 
 * @par Exemple d'utilisation :
 * @code
 * SchurSolver solver;
 * solver.SetSolverType(LinearSolverType::CG_ICHOL);
 * solver.SetTolerance(1e-10, 1000);
 * solver.SetMatrices(A, B, C);  // Prépare le solveur
 * 
 * Vec J, phi;
 * solver.Solve(rhs, J, phi);    // Résout le système
 * 
 * std::cout << "Itérations: " << solver.GetLastIterations() << std::endl;
 * @endcode
 */
class SchurSolver {
public:
    /**
     * @brief Constructeur par défaut
     * 
     * Initialise avec les paramètres par défaut :
     *   - Solveur : DIRECT_LU
     *   - Tolérance : 1e-10
     *   - Itérations max : 1000
     */
    SchurSolver();
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Définit le type de solveur linéaire
     * 
     * @param type Type de solveur (voir LinearSolverType)
     * 
     * @note Pour les matrices SPD (cas Σᵣ > 0), préférer CG_ICHOL ou DIRECT_LLT.
     *       Pour les matrices indéfinies, utiliser BICGSTAB ou DIRECT_LU.
     */
    void SetSolverType(LinearSolverType type) { solver_type_ = type; }
    
    /**
     * @brief Configure les paramètres de convergence (solveurs itératifs)
     * 
     * @param tol      Tolérance relative sur le résidu : ||r||/||b|| < tol
     * @param max_iter Nombre maximal d'itérations Krylov
     * 
     * @note Ces paramètres sont ignorés pour les solveurs directs.
     * 
     * @par Recommandations :
     *   - tol = 1e-10 pour problèmes critiques (keff)
     *   - tol = 1e-6 pour estimations rapides
     *   - max_iter = 2 × dimension pour systèmes bien conditionnés
     */
    void SetTolerance(double tol, int max_iter) { 
        tol_ = tol; 
        max_iter_ = max_iter; 
    }
    
    /**
     * @brief Active/désactive l'affichage des informations de convergence
     * 
     * @param v true pour activer les messages de diagnostic
     * 
     * En mode verbose, affiche pour chaque résolution :
     *   - Temps de calcul (ms)
     *   - Nombre d'itérations (solveurs itératifs)
     *   - Résidu final normalisé
     */
    void SetVerbose(bool v) { verbose_ = v; }
    
    // ========================================================================
    // RÉSOLUTION
    // ========================================================================
    
    /**
     * @brief Configure les matrices du système selle
     * 
     * @param A Matrice de masse RT (n_J × n_J, SPD)
     * @param B Matrice de divergence (n_φ × n_J)
     * @param C Matrice de réaction (n_φ × n_φ, SPD si Σᵣ > 0)
     * 
     * Cette méthode :
     * 1. Stocke les références aux matrices
     * 2. Calcule Bᵀ
     * 3. Factorise A (LU)
     * 4. Forme S et prépare le solveur si nécessaire
     * 
     * @throws std::runtime_error Si la factorisation de A échoue
     * 
     * @warning Les matrices doivent rester valides pendant toute la durée
     *          de vie du solveur. Aucune copie n'est effectuée.
     */
    void SetMatrices(const SpMat& A, const SpMat& B, const SpMat& C);
    
    /**
     * @brief Résout le système selle
     * 
     * @param[in]  rhs     Second membre f du système (dimension n_φ)
     * @param[out] J_sol   Solution courant J (dimension n_J)
     * @param[out] Phi_sol Solution flux φ (dimension n_φ)
     * 
     * Algorithme :
     * 1. Résoudre S·φ = f  (Schur explicite ou implicite)
     * 2. Calculer J = -A⁻¹·Bᵀ·φ
     * 
     * @throws std::runtime_error Si les matrices n'ont pas été configurées
     * 
     * @note Pour les solveurs itératifs, Phi_sol peut être utilisé comme
     *       estimation initiale (warm start) s'il est de bonne dimension.
     */
    void Solve(const Vec& rhs, Vec& J_sol, Vec& Phi_sol);
    
    // ========================================================================
    // DIAGNOSTICS
    // ========================================================================
    
    /**
     * @brief Retourne le nombre d'itérations de la dernière résolution
     * 
     * @return Nombre d'itérations (1 pour solveurs directs)
     */
    int GetLastIterations() const { return last_iterations_; }
    
    /**
     * @brief Retourne le résidu normalisé de la dernière résolution
     * 
     * @return ||S·φ - f|| / ||f||
     */
    double GetLastResidual() const { return last_residual_; }
    
private:
    // ========================================================================
    // MÉTHODES INTERNES
    // ========================================================================
    
    /**
     * @brief Vérifie si le solveur actuel est direct
     * @return true si DIRECT_LU, DIRECT_LDLT, ou DIRECT_LLT
     */
    bool IsDirectSolver() const;
    
    /**
     * @brief Détermine si S doit être formé explicitement
     * 
     * Critères :
     * - Solveur direct → toujours explicite
     * - Petits systèmes (n_φ < 200) → explicite
     * - Grands systèmes + itératif → implicite
     */
    bool NeedsExplicitSchur() const;
    
    /**
     * @brief Calcule S = C + B·A⁻¹·Bᵀ explicitement
     * 
     * Algorithme colonne par colonne :
     * 1. Pour j = 0..n_φ-1 :
     *    a. Extraire colonne j de Bᵀ : bⱼ = Bᵀ[:,j]
     *    b. Résoudre A·xⱼ = bⱼ
     *    c. Stocker xⱼ comme colonne de A⁻¹Bᵀ
     * 2. Calculer B·(A⁻¹Bᵀ)
     * 3. Ajouter C
     */
    void FormSchurComplement();
    
    /**
     * @brief Prépare le solveur pour S selon le type choisi
     * 
     * Pour chaque type, appelle compute() sur le solveur correspondant
     * avec la matrice S formée.
     */
    void PrepareSolver();
    
    /**
     * @brief Résout S·φ = f avec S explicite
     * 
     * Utilise le solveur préparé par PrepareSolver().
     */
    void SolveSchurExplicit(const Vec& rhs, Vec& phi);
    
    /**
     * @brief Résout S·φ = f avec produit Schur implicite (CG)
     * 
     * Implémente le gradient conjugué avec produit matrice-vecteur
     * calculé par SchurProduct().
     */
    void SolveSchurImplicit(const Vec& rhs, Vec& phi);
    
    /**
     * @brief Calcule le produit Schur implicite y = S·x
     * 
     * @param x Vecteur d'entrée (dimension n_φ)
     * @return y = (C + B·A⁻¹·Bᵀ)·x
     * 
     * Algorithme en 4 étapes :
     * 1. t₁ = Bᵀ·x           (produit creux)
     * 2. t₂ = A⁻¹·t₁         (substitution LU)
     * 3. t₃ = B·t₂           (produit creux)
     * 4. y = C·x + t₃        (produit creux + addition)
     */
    Vec SchurProduct(const Vec& x) const;
    
    // ========================================================================
    // DONNÉES MEMBRES
    // ========================================================================
    
    // Configuration
    LinearSolverType solver_type_;  ///< Type de solveur actuel
    double tol_;                    ///< Tolérance de convergence
    int max_iter_;                  ///< Itérations maximales
    bool verbose_;                  ///< Mode verbeux
    bool matrices_set_;             ///< Matrices configurées ?
    bool schur_formed_;             ///< S formé explicitement ?
    
    // Références aux matrices du système
    const SpMat* A_;    ///< Matrice de masse RT (non-owning)
    const SpMat* B_;    ///< Matrice de divergence (non-owning)
    const SpMat* C_;    ///< Matrice de réaction (non-owning)
    
    // Matrices dérivées
    SpMat BT_;          ///< Transposée de B
    SpMat S_;           ///< Complément de Schur (si explicite)
    
    // Solveur pour A (toujours LU pour la substitution)
    Eigen::SparseLU<SpMat> A_lu_solver_;
    
    // Solveurs pour S selon le type
    Eigen::SparseLU<SpMat> S_lu_;
    Eigen::SimplicialLDLT<SpMat> S_ldlt_;
    Eigen::SimplicialLLT<SpMat> S_llt_;
    
    // Solveurs itératifs avec différents préconditionneurs
    Eigen::ConjugateGradient<SpMat> cg_solver_;
    Eigen::ConjugateGradient<SpMat, Eigen::Lower, 
        Eigen::DiagonalPreconditioner<double>> cg_diag_solver_;
    Eigen::ConjugateGradient<SpMat, Eigen::Lower, 
        Eigen::IncompleteCholesky<double>> cg_ichol_solver_;
    Eigen::BiCGSTAB<SpMat> bicgstab_solver_;
    Eigen::BiCGSTAB<SpMat, 
        Eigen::DiagonalPreconditioner<double>> bicgstab_diag_solver_;
    Eigen::BiCGSTAB<SpMat, 
        Eigen::IncompleteLUT<double>> bicgstab_ilu_solver_;
    Eigen::LeastSquaresConjugateGradient<SpMat> lscg_solver_;
    
    // Statistiques de la dernière résolution
    int last_iterations_;    ///< Nombre d'itérations effectuées
    double last_residual_;   ///< Résidu final normalisé
};

// ============================================================================
// ACCÉLÉRATEUR DE CHEBYSHEV
// ============================================================================

/**
 * @class ChebyshevAccel
 * @brief Accélération de Chebyshev pour itérations de point fixe
 * 
 * Cette classe implémente l'accélération semi-itérative de Chebyshev pour
 * les schémas de puissance utilisés dans le calcul de k-effectif.
 * 
 * @section cheby_theory Fondement théorique
 * 
 * Pour une itération φ⁽ⁿ⁺¹⁾ = Gφ⁽ⁿ⁾ avec G = L⁻¹F (opérateur de puissance),
 * la convergence est gouvernée par le rapport de dominance :
 * 
 *     ρ = |λ₂|/|λ₁|
 * 
 * où λ₁ est la valeur propre dominante (keff) et λ₂ la suivante.
 * 
 * L'accélération de Chebyshev exploite les polynômes Tₙ(x) pour minimiser
 * l'erreur sur l'intervalle spectral [σ, 1] où σ ≈ 1 - 2(1-ρ).
 * 
 * @section cheby_formula Formule de récurrence
 * 
 * φ⁽ⁿ⁺¹⁾_acc = φ⁽ⁿ⁾_acc + aₙ(φ̃⁽ⁿ⁺¹⁾ - φ⁽ⁿ⁾_acc) + bₙ(φ⁽ⁿ⁾_acc - φ⁽ⁿ⁻¹⁾_acc)
 * 
 * avec φ̃⁽ⁿ⁺¹⁾ = Gφ⁽ⁿ⁾_acc (itération non accélérée)
 * 
 * Coefficients optimaux :
 *     aₙ = cosh((n-1)γ) / cosh(nγ)
 *     bₙ = cosh((n-2)γ) / cosh(nγ)
 *     γ = acosh(2/σ - 1)
 * 
 * @section cheby_usage Utilisation
 * 
 * @code
 * ChebyshevAccel accel(15, 0.98);  // nmax=15, sigma=0.98
 * 
 * for (int iter = 0; iter < max_iter; ++iter) {
 *     phi_new = solve_power_iteration(phi);  // Une itération
 *     accel(phi_new);                         // Accélère phi_new in-place
 *     
 *     if (converged(phi_new, phi)) break;
 *     phi = phi_new;
 * }
 * @endcode
 * 
 * @warning Le paramètre σ doit être estimé à partir du spectre de G.
 *          Une valeur trop petite peut causer des oscillations.
 */
class ChebyshevAccel {
public:
    /**
     * @brief Constructeur avec paramètres de l'accélérateur
     * 
     * @param nmax  Nombre maximal d'itérations avant réinitialisation
     * @param sigma Estimation du ratio spectral σ = λ_min/λ_max ∈ (0,1)
     * 
     * @note Typiquement σ ∈ [0.95, 0.99] pour les problèmes de réacteur.
     *       Une valeur de σ = 0.98 est un bon point de départ.
     */
    ChebyshevAccel(int nmax = 15, double sigma = 0.98);
    
    /**
     * @brief Destructeur (libère la mémoire des vecteurs historiques)
     */
    ~ChebyshevAccel();
    
    /**
     * @brief Applique l'accélération de Chebyshev
     * 
     * @param[in,out] phi Vecteur à accélérer (modifié sur place)
     * 
     * À l'itération n :
     * - n=0 : Stocke φ₀, aucune modification
     * - n=1 : Calcule φ₁_acc = φ₀ + a₁(φ₁ - φ₀)
     * - n≥2 : Applique la récurrence complète
     * 
     * L'accélérateur se réinitialise automatiquement après nmax itérations.
     */
    void operator()(Vec_t& phi);
    
    /**
     * @brief Réinitialise l'accélérateur
     * 
     * Efface l'historique et remet le compteur à zéro.
     * À appeler en cas de changement de problème ou de divergence.
     */
    void reset() { 
        m_it = 0; 
        delete m_phi0; m_phi0 = nullptr; 
        delete m_phi1; m_phi1 = nullptr; 
    }
    
private:
    int m_nmax;                     ///< Période de réinitialisation
    int m_it;                       ///< Compteur d'itérations actuel
    double m_sigma;                 ///< Ratio spectral estimé
    std::vector<double> m_coeffA;   ///< Coefficients aₙ pré-calculés
    std::vector<double> m_coeffB;   ///< Coefficients bₙ pré-calculés
    Vec_t* m_phi0;                  ///< φ⁽ⁿ⁻¹⁾
    Vec_t* m_phi1;                  ///< φ⁽ⁿ⁾
};

// ============================================================================
// ACCÉLÉRATEUR D'ANDERSON
// ============================================================================

/**
 * @class AndersonAccel
 * @brief Accélération d'Anderson (AA) pour itérations de point fixe
 * 
 * L'accélération d'Anderson est une méthode quasi-Newton qui utilise
 * l'historique des m derniers itérés pour extrapoler vers le point fixe.
 * 
 * @section anderson_theory Fondement théorique
 * 
 * Pour une itération x⁽ⁿ⁺¹⁾ = g(x⁽ⁿ⁾), on définit le résidu :
 *     f⁽ⁿ⁾ = g(x⁽ⁿ⁾) - x⁽ⁿ⁾
 * 
 * L'accélération cherche des coefficients α minimisant :
 *     ||Σᵢ αᵢ f⁽ⁿ⁻ᵢ⁾||²  sous contrainte  Σᵢ αᵢ = 1
 * 
 * Le nouvel itéré est :
 *     x⁽ⁿ⁺¹⁾_acc = Σᵢ αᵢ g(x⁽ⁿ⁻ᵢ⁾) = x⁽ⁿ⁾ + β · correction
 * 
 * @section anderson_algo Algorithme
 * 
 * 1. Stocker (x⁽ⁿ⁾, f⁽ⁿ⁾) dans l'historique circulaire
 * 2. Former la matrice F = [Δf⁽¹⁾ | Δf⁽²⁾ | ... | Δf⁽ᵐ⁾]
 * 3. Résoudre le problème de moindres carrés régularisé :
 *        (FᵀF + λI)α = Fᵀf⁽ⁿ⁾
 * 4. Calculer la correction : δx = Σᵢ αᵢ Δx⁽ⁱ⁾
 * 5. Appliquer le mixing : x_acc = x + β(x_cible - x)
 * 
 * @section anderson_params Paramètres
 * 
 * - m (profondeur) : Nombre d'itérés stockés. m=5 est un bon compromis.
 * - β (mixing) : Facteur de relaxation ∈ (0, 1]. β=1 = pas de relaxation.
 * - λ (régularisation) : Stabilise le système si mal conditionné.
 * 
 * @section anderson_vs_cheby Comparaison avec Chebyshev
 * 
 * | Critère          | Chebyshev           | Anderson            |
 * |------------------|---------------------|---------------------|
 * | A priori         | Nécessite σ estimé  | Aucun               |
 * | Mémoire          | O(n)                | O(m·n)              |
 * | Robustesse       | Sensible à σ        | Très robuste        |
 * | Optimalité       | Optimal si σ exact  | Quasi-optimal       |
 * 
 * @par Exemple d'utilisation :
 * @code
 * AndersonAccel accel(5, 1.0);  // m=5, beta=1.0
 * 
 * for (int iter = 0; iter < max_iter; ++iter) {
 *     phi_new = solve_iteration(phi);
 *     phi_new = accel(phi_new);  // Accélère et retourne le résultat
 *     
 *     if (converged(phi_new, phi)) break;
 *     phi = phi_new;
 * }
 * @endcode
 */
class AndersonAccel {
public:
    /**
     * @brief Constructeur avec paramètres
     * 
     * @param m    Profondeur de l'historique (défaut 5)
     * @param beta Paramètre de mixing (défaut 1.0 = pas de relaxation)
     */
    AndersonAccel(int m = 5, double beta = 1.0);
    
    /**
     * @brief Destructeur (libère les buffers circulaires)
     */
    ~AndersonAccel();
    
    /**
     * @brief Applique l'accélération d'Anderson
     * 
     * @param phi Vecteur courant (avant accélération)
     * @return Vecteur accéléré
     * 
     * @note Le vecteur d'entrée est utilisé pour calculer le résidu
     *       par rapport au dernier itéré stocké.
     */
    Vec_t operator()(Vec_t& phi);
    
    /**
     * @brief Réinitialise l'historique
     * 
     * À appeler en cas de changement de problème ou redémarrage.
     */
    void reset();
    
private:
    int m_max;                      ///< Profondeur maximale de l'historique
    double m_beta;                  ///< Paramètre de mixing
    double reg;                     ///< Régularisation de Tikhonov (λ)
    double max_rel;                 ///< Borne sur ||correction||/||x||
    
    std::deque<Vec_t> x_history;    ///< Historique des itérés x⁽ⁿ⁻ⁱ⁾
    std::deque<Vec_t> f_history;    ///< Historique des résidus f⁽ⁿ⁻ⁱ⁾
};

// ============================================================================
// SOLVEUR TRIDIAGONAL DE THOMAS
// ============================================================================

/**
 * @brief Résout un système tridiagonal par l'algorithme de Thomas
 * 
 * L'algorithme de Thomas (TDMA - TriDiagonal Matrix Algorithm) résout
 * le système Ax = b où A est tridiagonale en O(n) opérations.
 * 
 * @section thomas_matrix Structure de la matrice
 * 
 *     ┌                           ┐ ┌    ┐   ┌    ┐
 *     │ b₀  c₀   0   0  ...  0  0 │ │ x₀ │   │ d₀ │
 *     │ a₁  b₁  c₁   0  ...  0  0 │ │ x₁ │   │ d₁ │
 *     │  0  a₂  b₂  c₂  ...  0  0 │ │ x₂ │ = │ d₂ │
 *     │        ⋱   ⋱   ⋱          │ │  ⋮ │   │  ⋮ │
 *     │  0   0   0   0  ... aₙ bₙ │ │ xₙ │   │ dₙ │
 *     └                           ┘ └    ┘   └    ┘
 * 
 * @section thomas_algo Algorithme
 * 
 * Phase 1 - Élimination descendante :
 *     c'₀ = c₀/b₀
 *     c'ᵢ = cᵢ/(bᵢ - aᵢ·c'ᵢ₋₁)       pour i = 1..n-1
 *     
 *     d'₀ = d₀/b₀
 *     d'ᵢ = (dᵢ - aᵢ·d'ᵢ₋₁)/(bᵢ - aᵢ·c'ᵢ₋₁)
 * 
 * Phase 2 - Substitution remontante :
 *     xₙ = d'ₙ
 *     xᵢ = d'ᵢ - c'ᵢ·xᵢ₊₁           pour i = n-1..0
 * 
 * @param mat  Matrice tridiagonale creuse (RowMajor pour accès efficace)
 * @param b    Second membre
 * @param dest Vecteur solution (doit être pré-alloué)
 * 
 * @throws std::runtime_error Si les dimensions sont incompatibles
 * 
 * @note Complexité : O(n) en temps et O(n) en mémoire auxiliaire
 * @warning Instable si |bᵢ| < |aᵢ| + |cᵢ| (dominance diagonale non respectée)
 */
void ThomasSolver(const SpMatR_t& mat, const Vec_t& b, Vec_t& dest);

#endif // SOLVERS_HPP
