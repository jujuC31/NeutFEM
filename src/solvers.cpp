/**
 * @file solvers.cpp
 * @brief Implémentation des solveurs linéaires et accélérateurs
 * 
 * Ce fichier implémente les algorithmes de résolution pour le système selle
 * issu de la formulation mixte-duale RTₖ-Pₘ de l'équation de diffusion.
 * 
 * =============================================================================
 * STRATÉGIE DE RÉSOLUTION
 * =============================================================================
 * 
 * Le système selle mixte :
 * 
 *     ┌       ┐ ┌   ┐   ┌   ┐
 *     │ A  Bᵀ │ │ J │   │ 0 │
 *     │ B  C  │ │ φ │ = │ f │
 *     └       ┘ └   ┘   └   ┘
 * 
 * est résolu par élimination de J via le complément de Schur :
 * 
 *     S·φ = f   avec   S = C + B·A⁻¹·Bᵀ
 *     J = -A⁻¹·Bᵀ·φ
 * 
 * Deux approches selon la taille :
 * 
 * 1) PETITS SYSTÈMES (n_φ < seuil ou solveur direct) :
 *    - Former S explicitement colonne par colonne
 *    - Factoriser et résoudre directement
 * 
 * 2) GRANDS SYSTÈMES (n_φ ≥ seuil et solveur itératif) :
 *    - Ne PAS former S
 *    - Utiliser le produit implicite dans le solveur Krylov
 * 
 * =============================================================================
 * OPTIMISATIONS IMPLÉMENTÉES
 * =============================================================================
 * 
 * - Factorisation LU de A calculée une seule fois
 * - Produit Schur implicite pour éviter O(n²) de mémoire
 * - Seuillage adaptatif pour la formation de S
 * - Warm-start pour solveurs itératifs (BiCGSTAB)
 * - Mesures de performance intégrées
 * 
 * @author NeutFEM Development Team
 * @version 2.0
 * @date 2025
 */

#include "solvers.hpp"
#include <iomanip>
#include <chrono>
#include <cmath>
#include <iostream>

// ============================================================================
// SCHUR SOLVER - IMPLÉMENTATION
// ============================================================================

/**
 * @brief Constructeur par défaut du solveur Schur
 * 
 * Initialisation des paramètres :
 * - Solveur LU direct (robuste, usage général)
 * - Tolérance 1e-10 (précision standard neutronique)
 * - Maximum 1000 itérations (largement suffisant si bien conditionné)
 */
SchurSolver::SchurSolver()
    : solver_type_(LinearSolverType::DIRECT_LU)
    , tol_(1e-10)
    , max_iter_(1000)
    , verbose_(false)
    , matrices_set_(false)
    , schur_formed_(false)
    , last_iterations_(0)
    , last_residual_(0.0)
{}

/**
 * @brief Vérifie si le solveur configuré est direct
 * 
 * Les solveurs directs nécessitent la formation explicite de S car
 * ils doivent accéder à tous les éléments pour la factorisation.
 * 
 * @return true pour LU, LDLT, LLT
 */
bool SchurSolver::IsDirectSolver() const {
    return solver_type_ == LinearSolverType::DIRECT_LU ||
           solver_type_ == LinearSolverType::DIRECT_LDLT ||
           solver_type_ == LinearSolverType::DIRECT_LLT;
}

/**
 * @brief Détermine si S doit être formé explicitement
 * 
 * Heuristique de décision :
 * 
 * 1) Solveurs directs : TOUJOURS explicite
 *    - La factorisation nécessite tous les éléments
 *    - Pas d'alternative
 * 
 * 2) Matrice C non définie : TOUJOURS explicite
 *    - Cas dégénéré, ne devrait pas arriver en pratique
 * 
 * 3) Petits systèmes (n_φ < 200) : explicite
 *    - Le coût de formation est négligeable
 *    - La factorisation directe est plus rapide
 * 
 * 4) Grands systèmes + itératif : IMPLICITE
 *    - Économie de mémoire O(n²) → O(nnz)
 *    - Produit Schur implicite dans la boucle Krylov
 * 
 * @return true si S doit être calculé explicitement
 */
bool SchurSolver::NeedsExplicitSchur() const {
    // Solveurs directs : toujours explicite
    if (IsDirectSolver()) return true;
    
    // Sécurité : si C n'est pas défini
    if (C_ == nullptr) return true;
    
    // Seuil bas : seulement pour très petits systèmes
    // Au-delà, le produit implicite est plus efficace
    return C_->rows() < 200;
}

/**
 * @brief Configure les matrices du système selle
 * 
 * Séquence d'initialisation :
 * 
 * 1. Stockage des références (pas de copie)
 * 2. Calcul de Bᵀ (nécessaire pour le produit Schur)
 * 3. Factorisation LU de A
 *    - Utilisée pour A⁻¹ dans le produit Schur
 *    - Utilisée pour la reconstruction de J
 * 4. Formation de S si nécessaire (solveurs directs)
 * 
 * @param A Matrice de masse RT (n_J × n_J)
 * @param B Matrice de divergence (n_φ × n_J)
 * @param C Matrice de réaction (n_φ × n_φ)
 * 
 * @throws std::runtime_error Si A est singulière
 * 
 * @note Complexité :
 *       - Transposition de B : O(nnz(B))
 *       - Factorisation de A : O(n_J³) pire cas, O(nnz(A)^1.5) typique
 *       - Formation de S : O(n_φ · solve(A)) si nécessaire
 */
void SchurSolver::SetMatrices(const SpMat& A, const SpMat& B, const SpMat& C) {
    // Stockage des références (pas de copie des données)
    A_ = &A;
    B_ = &B;
    C_ = &C;
    
    // Calcul de Bᵀ pour le produit Schur
    // Eigen optimise cette opération pour les matrices creuses
    BT_ = B.transpose();
    
    // Factorisation LU de A (SparseLU avec pivotage)
    // Cette factorisation est réutilisée pour :
    //   1) Le produit Schur implicite : A⁻¹·v
    //   2) La reconstruction de J : J = -A⁻¹·Bᵀ·φ
    A_lu_solver_.compute(A);
    if (A_lu_solver_.info() != Eigen::Success) {
        throw std::runtime_error("Échec de la factorisation de A : "
            "matrice singulière ou mal conditionnée");
    }
    
    schur_formed_ = false;
    
    // Former S seulement si nécessaire
    // Pour les solveurs itératifs avec grands systèmes, on utilise
    // le produit implicite qui évite de stocker S
    if (NeedsExplicitSchur()) {
        FormSchurComplement();
    }
    
    matrices_set_ = true;
}

/**
 * @brief Résout le système selle complet
 * 
 * Algorithme en deux phases :
 * 
 * PHASE 1 - Résolution de S·φ = f :
 *   - Si S explicite : utilise le solveur préparé
 *   - Si S implicite : CG avec produit matrice-vecteur personnalisé
 * 
 * PHASE 2 - Reconstruction de J :
 *   J = -A⁻¹·Bᵀ·φ
 *   
 *   Cette formule vient de la première équation du système selle :
 *   A·J + Bᵀ·φ = 0  →  J = -A⁻¹·Bᵀ·φ
 * 
 * @param rhs     Second membre f (dimension n_φ)
 * @param J_sol   Solution courant (dimension n_J, sortie)
 * @param Phi_sol Solution flux (dimension n_φ, entrée/sortie)
 * 
 * @note Pour BiCGSTAB, Phi_sol est utilisé comme estimation initiale
 *       si sa dimension est correcte.
 */
void SchurSolver::Solve(const Vec& rhs, Vec& J_sol, Vec& Phi_sol) {
    if (!matrices_set_) {
        throw std::runtime_error("Matrices non configurées : "
            "appeler SetMatrices() avant Solve()");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Phase 1 : Résolution de S·φ = f
    if (NeedsExplicitSchur()) {
        // S est formé explicitement, utiliser le solveur préparé
        SolveSchurExplicit(rhs, Phi_sol);
    } else {
        // ✅ PRODUIT IMPLICITE - pas besoin de former S !
        // Utilise le gradient conjugué avec produit Schur personnalisé
        SolveSchurImplicit(rhs, Phi_sol);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Phase 2 : Reconstruction de J = -A⁻¹·Bᵀ·φ
    // Étapes :
    //   1. tmp = Bᵀ·φ     (produit matrice-vecteur)
    //   2. J = -A⁻¹·tmp   (substitution avant/arrière avec facteurs LU)
    Vec tmp = BT_ * Phi_sol;
    J_sol = -A_lu_solver_.solve(tmp);
    
    // Affichage des statistiques si mode verbeux
    if (verbose_) {
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "    Schur: " << elapsed << " ms";
        if (!IsDirectSolver()) {
            std::cout << ", " << last_iterations_ << " it, res=" 
                      << std::scientific << std::setprecision(2) << last_residual_;
        }
        std::cout << std::defaultfloat << std::endl;
    }
}

/**
 * @brief Forme le complément de Schur S = C + B·A⁻¹·Bᵀ explicitement
 * 
 * Algorithme colonne par colonne pour minimiser la mémoire temporaire :
 * 
 * Pour j = 0, 1, ..., n_φ - 1 :
 *   1. Extraire la colonne j de Bᵀ : bⱼ = Bᵀ[:, j]
 *   2. Résoudre le système : A·xⱼ = bⱼ
 *   3. Stocker xⱼ comme colonne j de la matrice (A⁻¹·Bᵀ)
 * 
 * Puis calculer : S = C + B · (A⁻¹·Bᵀ)
 * 
 * @note Le seuillage à 1e-14 évite le remplissage inutile de S
 *       avec des valeurs numériquement nulles.
 * 
 * @note Complexité : O(n_φ × coût_solve(A) + nnz(B) × n_J)
 */
void SchurSolver::FormSchurComplement() {
    int n_Phi = C_->rows();
    int n_J = A_->rows();
    
    if (verbose_) {
        std::cout << "    Forming Schur complement (" << n_Phi << " cols)..." << std::flush;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // ========================================================================
    // Calcul de A⁻¹·Bᵀ colonne par colonne
    // ========================================================================
    // 
    // On évite de former A⁻¹ explicitement (dense et coûteux).
    // À la place, on résout A·x = b pour chaque colonne de Bᵀ.
    
    SpMat AinvBT(n_J, n_Phi);
    std::vector<Triplet> triplets;
    triplets.reserve(n_J * 10);  // Estimation du remplissage
    
    for (int j = 0; j < n_Phi; ++j) {
        // Extraire la colonne j de Bᵀ
        Vec col_BT = BT_.col(j);
        
        // Résoudre A·x = Bᵀ[:, j] par substitution LU
        Vec col_AinvBT = A_lu_solver_.solve(col_BT);
        
        // Stocker les éléments non-nuls (seuillage pour éviter fill-in)
        for (int i = 0; i < n_J; ++i) {
            if (std::abs(col_AinvBT(i)) > 1e-14) {
                triplets.emplace_back(i, j, col_AinvBT(i));
            }
        }
    }
    AinvBT.setFromTriplets(triplets.begin(), triplets.end());
    AinvBT.makeCompressed();
    
    // ========================================================================
    // Formation de S = C + B·(A⁻¹·Bᵀ)
    // ========================================================================
    SpMat BA = ((*B_) * AinvBT).eval();  // .eval() force l'évaluation
    S_ = *C_ + BA;
    S_.makeCompressed();  // Optimise le stockage
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (verbose_) {
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << " done (" << elapsed << " ms)" << std::endl;
    }
    
    // Préparer le solveur pour S
    PrepareSolver();
    schur_formed_ = true;
}

/**
 * @brief Prépare le solveur linéaire pour la matrice S
 * 
 * Selon le type de solveur configuré, cette méthode :
 * 
 * - DIRECT_* : Effectue la factorisation complète de S
 * - ITÉRATIF : Configure les paramètres et éventuellement le préconditionneur
 * 
 * @note La factorisation est la partie la plus coûteuse pour les
 *       solveurs directs. Pour les itératifs, compute() prépare
 *       seulement le préconditionneur.
 */
void SchurSolver::PrepareSolver() {
    switch (solver_type_) {
        // ====================================================================
        // SOLVEURS DIRECTS
        // ====================================================================
        
        case LinearSolverType::DIRECT_LU:
            // Factorisation LU avec pivotage partiel
            // Usage général, robuste pour toute matrice inversible
            S_lu_.compute(S_);
            break;
            
        case LinearSolverType::DIRECT_LDLT:
            // Factorisation LDLᵀ pour matrices symétriques
            // 2× plus rapide que LU, accepte les matrices indéfinies
            S_ldlt_.compute(S_);
            break;
            
        case LinearSolverType::DIRECT_LLT:
            // Factorisation de Cholesky LLᵀ
            // Le plus rapide, mais UNIQUEMENT pour matrices SPD
            S_llt_.compute(S_);
            break;
        
        // ====================================================================
        // SOLVEURS ITÉRATIFS DE TYPE GRADIENT CONJUGUÉ
        // (pour matrices symétriques définies positives)
        // ====================================================================
        
        case LinearSolverType::CG:
            // CG sans préconditionnement
            // Simple mais convergence lente si κ(S) élevé
            cg_solver_.setMaxIterations(max_iter_);
            cg_solver_.setTolerance(tol_);
            cg_solver_.compute(S_);
            break;
            
        case LinearSolverType::CG_DIAG:
            // CG + préconditionneur diagonal (Jacobi)
            // M = diag(S), coût négligeable, améliore κ
            cg_diag_solver_.setMaxIterations(max_iter_);
            cg_diag_solver_.setTolerance(tol_);
            cg_diag_solver_.compute(S_);
            break;
            
        case LinearSolverType::CG_ICHOL:
            // CG + Cholesky incomplet
            // Excellent pour problèmes de diffusion (matrice M-matrix)
            // Préconditionneur optimal en terme efficacité/coût
            cg_ichol_solver_.setMaxIterations(max_iter_);
            cg_ichol_solver_.setTolerance(tol_);
            cg_ichol_solver_.compute(S_);
            break;
        
        // ====================================================================
        // SOLVEURS ITÉRATIFS DE TYPE BICGSTAB
        // (pour matrices non-symétriques ou mal conditionnées)
        // ====================================================================
        
        case LinearSolverType::BICGSTAB:
            // BiCGSTAB sans préconditionnement
            // Robuste pour matrices non-symétriques
            bicgstab_solver_.setMaxIterations(max_iter_);
            bicgstab_solver_.setTolerance(tol_);
            bicgstab_solver_.compute(S_);
            break;
            
        case LinearSolverType::BICGSTAB_DIAG:
            // BiCGSTAB + préconditionneur diagonal
            bicgstab_diag_solver_.setMaxIterations(max_iter_);
            bicgstab_diag_solver_.setTolerance(tol_);
            bicgstab_diag_solver_.compute(S_);
            break;
            
        case LinearSolverType::BICGSTAB_ILU:
            // BiCGSTAB + factorisation ILU incomplète
            // Le plus robuste pour matrices difficiles
            bicgstab_ilu_solver_.setMaxIterations(max_iter_);
            bicgstab_ilu_solver_.setTolerance(tol_);
            bicgstab_ilu_solver_.compute(S_);
            break;
        
        // ====================================================================
        // SOLVEUR MOINDRES CARRÉS
        // ====================================================================
        
        case LinearSolverType::LCG:
            // Least Squares Conjugate Gradient
            // Pour systèmes sur/sous-déterminés ou singuliers
            lscg_solver_.setMaxIterations(max_iter_);
            lscg_solver_.setTolerance(tol_);
            lscg_solver_.compute(S_);
            break;
            
        default:
            // Fallback sur LU si type inconnu
            S_lu_.compute(S_);
            break;
    }
}

/**
 * @brief Résout S·φ = f avec S formé explicitement
 * 
 * Dispatch vers le solveur approprié selon le type configuré.
 * 
 * @param rhs Second membre f
 * @param phi Solution φ (sortie, peut servir d'estimation initiale pour BiCGSTAB)
 */
void SchurSolver::SolveSchurExplicit(const Vec& rhs, Vec& phi) {
    switch (solver_type_) {
        // Solveurs directs : une seule "itération"
        case LinearSolverType::DIRECT_LU:
            phi = S_lu_.solve(rhs);
            last_iterations_ = 1;
            break;
            
        case LinearSolverType::DIRECT_LDLT:
            phi = S_ldlt_.solve(rhs);
            last_iterations_ = 1;
            break;
            
        case LinearSolverType::DIRECT_LLT:
            phi = S_llt_.solve(rhs);
            last_iterations_ = 1;
            break;
        
        // Solveurs CG : estimation initiale x₀ = 0 par défaut
        case LinearSolverType::CG:
            phi = cg_solver_.solve(rhs);
            last_iterations_ = cg_solver_.iterations();
            last_residual_ = cg_solver_.error();
            break;
            
        case LinearSolverType::CG_DIAG:
            phi = cg_diag_solver_.solve(rhs);
            last_iterations_ = cg_diag_solver_.iterations();
            last_residual_ = cg_diag_solver_.error();
            break;
            
        case LinearSolverType::CG_ICHOL:
            phi = cg_ichol_solver_.solve(rhs);
            last_iterations_ = cg_ichol_solver_.iterations();
            last_residual_ = cg_ichol_solver_.error();
            break;
        
        // Solveurs BiCGSTAB : utilisent phi comme estimation initiale (warm start)
        case LinearSolverType::BICGSTAB:
            phi = bicgstab_solver_.solveWithGuess(rhs, phi);
            last_iterations_ = bicgstab_solver_.iterations();
            last_residual_ = bicgstab_solver_.error();
            break;
            
        case LinearSolverType::BICGSTAB_DIAG:
            phi = bicgstab_diag_solver_.solveWithGuess(rhs, phi);
            last_iterations_ = bicgstab_diag_solver_.iterations();
            last_residual_ = bicgstab_diag_solver_.error();
            break;
            
        case LinearSolverType::BICGSTAB_ILU:
            phi = bicgstab_ilu_solver_.solveWithGuess(rhs, phi);
            last_iterations_ = bicgstab_ilu_solver_.iterations();
            last_residual_ = bicgstab_ilu_solver_.error();
            break;
        
        case LinearSolverType::LCG:
            phi = lscg_solver_.solve(rhs);
            last_iterations_ = lscg_solver_.iterations();
            last_residual_ = lscg_solver_.error();
            break;
            
        default:
            phi = S_lu_.solve(rhs);
            last_iterations_ = 1;
            break;
    }
    
    // Pour les solveurs directs, calculer le résidu a posteriori
    if (IsDirectSolver()) {
        last_residual_ = (S_ * phi - rhs).norm() / rhs.norm();
    }
}

/**
 * @brief Calcule le produit Schur IMPLICITE : y = S·x = (C + B·A⁻¹·Bᵀ)·x
 * 
 * Cette fonction est le cœur de l'approche sans formation de S.
 * Elle calcule le produit matrice-vecteur en 4 étapes :
 * 
 *     t₁ = Bᵀ·x           O(nnz(B))   produit creux
 *     t₂ = A⁻¹·t₁         O(n_J)      substitution LU (facteurs pré-calculés)
 *     t₃ = B·t₂           O(nnz(B))   produit creux
 *     y  = C·x + t₃       O(nnz(C))   produit creux + addition
 * 
 * Complexité totale : O(nnz(A) + 2·nnz(B) + nnz(C)) par appel
 * 
 * Comparaison avec formation explicite :
 * - Formation de S : O(n_φ · solve(A)) une fois
 * - Produit S·x explicite : O(nnz(S))
 * - Produit implicite : O(nnz(A) + 2·nnz(B) + nnz(C))
 * 
 * Pour les grands systèmes où nnz(S) >> nnz(A,B,C), le produit
 * implicite est plus efficace en mémoire ET en temps.
 * 
 * @param x Vecteur d'entrée (dimension n_φ)
 * @return y = S·x
 */
Vec SchurSolver::SchurProduct(const Vec& x) const {
    // Étape 1 : t₁ = Bᵀ·x
    Vec t1 = BT_ * x;
    
    // Étape 2 : t₂ = A⁻¹·t₁ (substitution avec facteurs LU pré-calculés)
    Vec t2 = A_lu_solver_.solve(t1);
    
    // Étape 3 : t₃ = B·t₂
    Vec t3 = (*B_) * t2;
    
    // Étape 4 : y = C·x + t₃
    return (*C_) * x + t3;
}

/**
 * @brief Résout S·φ = f avec produit Schur implicite (Gradient Conjugué)
 * 
 * Implémentation du gradient conjugué préconditionné (ici sans préconditionneur
 * pour simplicité) avec produit matrice-vecteur calculé par SchurProduct().
 * 
 * ALGORITHME CG :
 * 
 * Initialisation :
 *     x₀ = 0
 *     r₀ = b - S·x₀ = b
 *     p₀ = r₀
 * 
 * Pour k = 0, 1, 2, ... jusqu'à convergence :
 *     αₖ = (rₖᵀrₖ) / (pₖᵀS·pₖ)        // Pas optimal dans la direction pₖ
 *     xₖ₊₁ = xₖ + αₖpₖ                 // Mise à jour de la solution
 *     rₖ₊₁ = rₖ - αₖS·pₖ               // Mise à jour du résidu
 *     βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)      // Coefficient de conjugaison
 *     pₖ₊₁ = rₖ₊₁ + βₖpₖ               // Nouvelle direction de recherche
 * 
 * PROPRIÉTÉS :
 * - Convergence garantie en n itérations pour matrices SPD
 * - Convergence pratique en O(√κ) itérations
 * - Robuste numériquement pour matrices bien conditionnées
 * 
 * @param rhs Second membre f
 * @param phi Solution φ (sortie)
 */
void SchurSolver::SolveSchurImplicit(const Vec& rhs, Vec& phi) {
    const int n = rhs.size();
    
    // ========================================================================
    // INITIALISATION
    // ========================================================================
    phi = Vec::Zero(n);  // x₀ = 0
    Vec r = rhs;         // r₀ = b - S·x₀ = b (car x₀ = 0)
    Vec p = r;           // p₀ = r₀ (direction initiale = gradient)
    
    double r_dot_r = r.squaredNorm();  // ||r₀||²
    const double rhs_norm = rhs.norm();
    const double tol_sq = tol_ * tol_ * rhs_norm * rhs_norm;  // Critère d'arrêt
    
    last_iterations_ = 0;
    
    // ========================================================================
    // BOUCLE PRINCIPALE DU GRADIENT CONJUGUÉ
    // ========================================================================
    for (int k = 0; k < max_iter_; ++k) {
        // Produit Schur IMPLICITE : Ap = S·p
        // C'est ici que réside l'avantage : on n'a jamais formé S
        Vec Ap = SchurProduct(p);
        
        // Calcul du pas optimal αₖ = (rₖᵀrₖ) / (pₖᵀA·pₖ)
        double p_dot_Ap = p.dot(Ap);
        
        // Protection contre division par zéro (direction nulle ou breakdown)
        if (std::abs(p_dot_Ap) < 1e-30) break;
        
        double alpha = r_dot_r / p_dot_Ap;
        
        // Mise à jour de la solution : xₖ₊₁ = xₖ + αₖpₖ
        phi += alpha * p;
        
        // Mise à jour du résidu : rₖ₊₁ = rₖ - αₖA·pₖ
        r -= alpha * Ap;
        
        double r_dot_r_new = r.squaredNorm();
        last_iterations_ = k + 1;
        
        // Test de convergence : ||rₖ₊₁|| / ||b|| < tol
        if (r_dot_r_new < tol_sq) {
            last_residual_ = std::sqrt(r_dot_r_new) / rhs_norm;
            return;
        }
        
        // Coefficient de conjugaison : βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)
        double beta = r_dot_r_new / r_dot_r;
        
        // Nouvelle direction : pₖ₊₁ = rₖ₊₁ + βₖpₖ
        p = r + beta * p;
        
        // Préparer l'itération suivante
        r_dot_r = r_dot_r_new;
    }
    
    // Si on arrive ici, max_iter atteint sans convergence
    last_residual_ = std::sqrt(r_dot_r) / rhs_norm;
}

// ============================================================================
// ACCÉLÉRATEUR DE CHEBYSHEV - IMPLÉMENTATION
// ============================================================================

/**
 * @brief Constructeur de l'accélérateur de Chebyshev
 * 
 * Pré-calcule les coefficients d'accélération basés sur les polynômes
 * de Tchebychev de première espèce.
 * 
 * THÉORIE :
 * 
 * Pour une itération xₙ₊₁ = Gxₙ avec spectre(G) ⊂ [σ, 1], les polynômes
 * de Tchebychev minimisent ||Tₙ(G)||_∞ sur [σ, 1].
 * 
 * Les coefficients de la récurrence sont :
 *     γ = acosh(2/σ - 1)
 *     aₙ = cosh((n-1)γ) / cosh(nγ)
 *     bₙ = cosh((n-2)γ) / cosh(nγ)
 * 
 * avec la récurrence accélérée :
 *     φₙ₊₁ = φₙ + aₙ(φ̃ₙ₊₁ - φₙ) + bₙ(φₙ - φₙ₋₁)
 * 
 * @param nmax  Nombre d'itérations avant réinitialisation
 * @param sigma Ratio spectral estimé ∈ (0, 1)
 */
ChebyshevAccel::ChebyshevAccel(int nmax, double sigma) 
    : m_nmax(nmax), m_it(0), m_sigma(sigma) {
    
    // Pré-allocation des tableaux de coefficients
    m_coeffA.resize(nmax);
    m_coeffB.resize(nmax);
    
    // Paramètre γ de la transformation de Tchebychev
    // γ = acosh(2/σ - 1) mappe [σ, 1] vers [1, ∞) via cosh
    double G = acosh(2. / m_sigma - 1.);
    
    // Coefficients non utilisés pour n=0
    m_coeffA[0] = 0.;
    m_coeffB[0] = 0.;
    
    // Coefficient pour n=1 (première itération accélérée)
    // a₁ = 2 / (2 - σ) simplifié
    m_coeffA[1] = 2. / (2. - m_sigma);
    m_coeffB[1] = 0.;  // Pas de terme (φₙ - φₙ₋₁) pour n=1
    
    // Coefficients pour n ≥ 2
    for(int k = 2; k < nmax; ++k) {
        m_coeffA[k] = cosh((k-1) * G) / cosh(k * G);
        m_coeffB[k] = cosh((k-2) * G) / cosh(k * G);
    }
    
    // Initialisation des pointeurs d'historique
    m_phi0 = nullptr;
    m_phi1 = nullptr;
}

/**
 * @brief Destructeur - libère la mémoire des vecteurs historiques
 */
ChebyshevAccel::~ChebyshevAccel() {
    delete m_phi0;
    delete m_phi1;
}

/**
 * @brief Applique l'accélération de Chebyshev à un vecteur
 * 
 * COMPORTEMENT SELON L'ITÉRATION :
 * 
 * m_it = 0 : Stocke φ₀ comme référence initiale
 * m_it = 1 : Première accélération : φ₁_acc = φ₀ + a₁(φ₁ - φ₀)
 * m_it ≥ 2 : Récurrence complète :
 *            φₙ_acc = φₙ₋₁_acc + (4/σ)·aₙ·(φ̃ₙ - φₙ₋₁_acc) + bₙ·(φₙ₋₁_acc - φₙ₋₂_acc)
 * 
 * Après nmax itérations, l'accélérateur se réinitialise automatiquement.
 * 
 * @param phi Vecteur à accélérer (modifié sur place)
 * 
 * @note Le facteur 4/σ dans la récurrence vient de la normalisation
 *       pour que l'itération non-accélérée corresponde à G.
 */
void ChebyshevAccel::operator()(Vec_t &phi) {
    // Réinitialisation périodique pour éviter l'accumulation d'erreurs
    if(m_it == m_nmax) {
        m_it = 0;
        delete m_phi0; m_phi0 = nullptr;
        delete m_phi1; m_phi1 = nullptr;
    }

    if(m_it == 0) {
        // Première itération : simplement stocker φ₀
        m_phi0 = new Vec_t(phi);
        ++m_it;
    } 
    else if(m_it == 1) {
        // Deuxième itération : première accélération
        // φ₁_acc = φ₀ + a₁·(φ₁ - φ₀)
        m_phi1 = new Vec_t(phi.size());
        (*m_phi1) = (*m_phi0) + m_coeffA[1] * (phi - (*m_phi0));
        phi = (*m_phi1);
        ++m_it;
    } 
    else {
        // Itérations suivantes : récurrence complète
        // φₙ_acc = φₙ₋₁_acc + (4/σ)·aₙ·(φ̃ₙ - φₙ₋₁_acc) + bₙ·(φₙ₋₁_acc - φₙ₋₂_acc)
        Vec_t *new_phi = new Vec_t(phi.size());
        (*new_phi) = (*m_phi1) 
                   + (4. / m_sigma) * m_coeffA[m_it] * (phi - (*m_phi1))
                   + m_coeffB[m_it] * ((*m_phi1) - (*m_phi0));
        
        // Décaler l'historique : φₙ₋₂ ← φₙ₋₁, φₙ₋₁ ← φₙ
        delete m_phi0;
        m_phi0 = m_phi1;
        m_phi1 = new_phi;
        phi = (*new_phi);
        ++m_it;
    }
}

// ============================================================================
// ACCÉLÉRATEUR D'ANDERSON - IMPLÉMENTATION
// ============================================================================

/**
 * @brief Constructeur de l'accélérateur d'Anderson
 * 
 * @param m    Profondeur de l'historique (nombre d'itérés stockés)
 * @param beta Paramètre de mixing (1.0 = pas de relaxation)
 * 
 * Paramètres internes :
 * - reg = 1e-8 : régularisation de Tikhonov pour stabilité numérique
 * - max_rel = 0.3 : borne sur ||correction||/||x|| pour éviter divergence
 */
AndersonAccel::AndersonAccel(int m, double beta)
    : m_max(m), m_beta(beta), reg(1e-8), max_rel(0.3) {}

/**
 * @brief Destructeur - nettoie les buffers circulaires
 */
AndersonAccel::~AndersonAccel() {
    x_history.clear();
    f_history.clear();
}

/**
 * @brief Réinitialise l'historique de l'accélérateur
 */
void AndersonAccel::reset() {
    x_history.clear();
    f_history.clear();
}

/**
 * @brief Applique l'accélération d'Anderson
 * 
 * ALGORITHME :
 * 
 * 1. Calcul du résidu : f⁽ⁿ⁾ = g(x⁽ⁿ⁾) - x⁽ⁿ⁾
 *    (phi est g(x_old), donc f_new = phi - x_old)
 * 
 * 2. Mise à jour de l'historique (structure FIFO de profondeur m)
 * 
 * 3. Construction du système de moindres carrés :
 *    - F = [Δf⁽¹⁾ | ... | Δf⁽ᵐ⁻¹⁾] avec Δf⁽ⁱ⁾ = f⁽ⁱ⁺¹⁾ - f⁽ⁱ⁾
 *    - Minimiser ||F·α - f⁽ⁿ⁾ + f⁽ⁿ⁻¹⁾||²
 * 
 * 4. Résolution régularisée : (FᵀF + λI)α = Fᵀ(f⁽ⁿ⁾ - f⁽ⁿ⁻¹⁾)
 * 
 * 5. Calcul de la correction : δx = Σᵢ αᵢ·Δx⁽ⁱ⁾
 * 
 * 6. Application avec safeguard et mixing :
 *    x_acc = (1-β)·x + β·(x - δx_clipped)
 * 
 * @param phi Vecteur courant (après une itération non accélérée)
 * @return Vecteur accéléré
 */
Vec_t AndersonAccel::operator()(Vec_t& phi) {
    // Première itération : initialiser l'historique
    if (x_history.empty()) {
        x_history.push_back(phi);
        f_history.push_back(Vec_t::Zero(phi.size()));
        return phi;
    }

    // Récupérer le dernier itéré et calculer le résidu
    const Vec_t& x_old = x_history.back();
    Vec_t f_new = phi - x_old;  // Résidu = g(x) - x

    // Ajouter à l'historique
    x_history.push_back(phi);
    f_history.push_back(f_new);
    
    // Maintenir la taille de l'historique (structure FIFO)
    if (static_cast<int>(x_history.size()) > m_max) {
        x_history.pop_front();
        f_history.pop_front();
    }

    int m = static_cast<int>(f_history.size());
    
    // Pas assez d'historique pour l'extrapolation
    if (m == 1) return phi;

    // ========================================================================
    // Construction de la matrice F des différences de résidus
    // F[:, i] = f⁽ⁱ⁺¹⁾ - f⁽ⁱ⁾
    // ========================================================================
    Eigen::MatrixXd F(phi.size(), m - 1);
    for (int i = 0; i < m - 1; ++i) {
        F.col(i) = f_history[i + 1] - f_history[i];
    }
    
    // Second membre : f⁽ⁿ⁾ - f⁽ⁿ⁻¹⁾
    Vec_t rhs = f_new - f_history[m - 2];

    // ========================================================================
    // Système normal régularisé : (FᵀF + λI)α = Fᵀ·rhs
    // ========================================================================
    Eigen::MatrixXd A = F.transpose() * F;
    Eigen::VectorXd b = F.transpose() * rhs;

    // Régularisation de Tikhonov pour stabilité
    for (int i = 0; i < A.rows(); ++i) {
        A(i, i) += reg;
    }

    // Résolution par factorisation LDLT (symétrique)
    Eigen::VectorXd alpha = A.ldlt().solve(b);

    // ========================================================================
    // Calcul de la correction : δx = Σᵢ αᵢ·Δx⁽ⁱ⁾
    // ========================================================================
    Vec_t delta_x = Vec_t::Zero(phi.size());
    for (int i = 0; i < m - 1; ++i) {
        delta_x += alpha(i) * (x_history[i + 1] - x_history[i]);
    }

    // ========================================================================
    // Safeguard : limiter la correction relative
    // ========================================================================
    double phi_norm = phi.norm();
    double dx_norm = delta_x.norm();
    if (phi_norm > 0 && dx_norm / phi_norm > max_rel) {
        // Réduire la correction pour éviter les grandes oscillations
        delta_x *= (max_rel * phi_norm / dx_norm);
    }

    // ========================================================================
    // Application du mixing : x_acc = (1-β)·x + β·(x - δx)
    // ========================================================================
    Vec_t x_accel = phi - delta_x;
    return (1.0 - m_beta) * phi + m_beta * x_accel;
}

// ============================================================================
// SOLVEUR TRIDIAGONAL DE THOMAS - IMPLÉMENTATION
// ============================================================================

/**
 * @brief Résout un système tridiagonal Ax = b par l'algorithme de Thomas
 * 
 * L'algorithme de Thomas (ou TDMA) est une version spécialisée de
 * l'élimination de Gauss pour les matrices tridiagonales.
 * 
 * STRUCTURE DE LA MATRICE :
 * 
 *     ┌                                   ┐
 *     │ b₀  c₀   0   0   ...  0    0   0  │
 *     │ a₁  b₁  c₁   0   ...  0    0   0  │
 *     │  0  a₂  b₂  c₂   ...  0    0   0  │
 *     │            ...                     │
 *     │  0   0   0   0   ... aₙ₋₁ bₙ₋₁ cₙ₋₁│
 *     │  0   0   0   0   ...  0   aₙ   bₙ │
 *     └                                   ┘
 * 
 * ALGORITHME EN DEUX PHASES :
 * 
 * Phase 1 - Élimination descendante (forward sweep) :
 * Transforme le système en forme triangulaire supérieure.
 * 
 *     c'₀ = c₀ / b₀
 *     c'ᵢ = cᵢ / (bᵢ - aᵢ·c'ᵢ₋₁)        pour i = 1..n-1
 *     
 *     d'₀ = d₀ / b₀
 *     d'ᵢ = (dᵢ - aᵢ·d'ᵢ₋₁) / (bᵢ - aᵢ·c'ᵢ₋₁)
 * 
 * Phase 2 - Substitution remontante (back substitution) :
 * 
 *     xₙ = d'ₙ
 *     xᵢ = d'ᵢ - c'ᵢ·xᵢ₊₁               pour i = n-1..0
 * 
 * COMPLEXITÉ : O(n) en temps, O(n) en espace auxiliaire
 * 
 * STABILITÉ : Stable si |bᵢ| ≥ |aᵢ| + |cᵢ| (dominance diagonale stricte)
 * 
 * @param mat  Matrice tridiagonale en format RowMajor
 * @param b    Vecteur second membre
 * @param dest Vecteur solution (pré-alloué)
 * 
 * @throws std::runtime_error Si les dimensions sont incompatibles
 */
void ThomasSolver(const SpMatR_t& mat, const Vec_t& b, Vec_t& dest) {
    // Vérification des dimensions
    if ((b.rows() != mat.rows()) || (b.rows() != mat.cols()) ||
        (dest.rows() != b.rows())) {
        throw std::runtime_error("ThomasSolver: dimensions incorrectes - "
            "la matrice doit être carrée et les vecteurs de même taille");
    }
    
    long N = b.rows();

    // ========================================================================
    // Phase 1a : Calcul des coefficients c' modifiés
    // ========================================================================
    Vec_t C(N);
    
    // Première ligne : c'₀ = c₀ / b₀
    C(0) = mat.coeff(0, 1) / mat.coeff(0, 0);
    
    // Lignes suivantes : c'ᵢ = cᵢ / (bᵢ - aᵢ·c'ᵢ₋₁)
    for (long i = 1; i < N - 1; i++) {
        double denominator = mat.coeff(i, i) - mat.coeff(i, i - 1) * C(i - 1);
        C(i) = mat.coeff(i, i + 1) / denominator;
    }

    // ========================================================================
    // Phase 1b : Calcul du second membre modifié d'
    // ========================================================================
    Vec_t D(N);
    
    // Première ligne : d'₀ = d₀ / b₀
    D(0) = b(0) / mat.coeff(0, 0);
    
    // Lignes suivantes : d'ᵢ = (dᵢ - aᵢ·d'ᵢ₋₁) / (bᵢ - aᵢ·c'ᵢ₋₁)
    for (long i = 1; i < N; i++) {
        double denominator = mat.coeff(i, i) - mat.coeff(i, i - 1) * C(i - 1);
        D(i) = (b(i) - mat.coeff(i, i - 1) * D(i - 1)) / denominator;
    }

    // ========================================================================
    // Phase 2 : Substitution remontante
    // ========================================================================
    
    // Dernière inconnue : xₙ = d'ₙ
    dest(N - 1) = D(N - 1);
    
    // Remontée : xᵢ = d'ᵢ - c'ᵢ·xᵢ₊₁
    for (long i = N - 2; i >= 0; i--) {
        dest(i) = D(i) - C(i) * dest(i + 1);
    }
}
