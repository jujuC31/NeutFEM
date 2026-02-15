/**
 * @file NeutFEM.cpp
 * @brief Solveur de diffusion neutronique multi-groupes - Formulation mixte RTₖ-Pₖ
 * 
 * @details
 * Ce module implémente un solveur d'équation de diffusion neutronique en 
 * formulation variationnelle mixte utilisant les éléments de Raviart-Thomas (RTₖ)
 * pour le courant neutronique et les polynômes de Legendre discontinus (Pₖ)
 * pour le flux scalaire.
 * 
 * ARCHITECTURE MATHÉMATIQUE:
 * -------------------------
 * Équation de diffusion multigroupe avec conditions aux limites:
 *   -∇·(D_g ∇φ_g) + Σ_r,g φ_g = χ_g/k × Σ_{g'} νΣ_{f,g'} φ_{g'} 
 *                              + Σ_{g'≠g} Σ_{s,g'→g} φ_{g'}
 * 
 * Formulation mixte-duale (Hébert):
 *   J = -D ∇φ           (loi de Fick)
 *   ∇·J + Σ_r φ = S     (bilan neutronique)
 * 
 * Système variationnel:
 *   (1/D) <J, ψ> + <φ, ∇·ψ> = <φ_bord, ψ·n>_∂Ω   ∀ψ ∈ H(div)
 *   <∇·J, v> + <Σ_r φ, v> = <S, v>               ∀v ∈ L²
 * 
 * ALGORITHME DE RÉSOLUTION:
 * ------------------------
 * 1. Assemblage des matrices élémentaires par quadrature de Gauss-Legendre
 * 2. Élimination du courant J par complément de Schur: S = C + B A⁻¹ Bᵀ
 * 3. Itérations de puissance avec accélération de Chebyshev/CMFD pour k-eff
 * 4. Balayage Gauss-Seidel sur les groupes d'énergie
 * 
 * OPTIMISATIONS POUR ACCELERATION TCPU:
 * ----------------
 * - Cache de factorisation Schur par groupe (évite re-factorisation)
 * - Solveur diagonal pour RT0-P0 (inversion triviale)
 * - Accélération CMFD (Coarse Mesh Finite Difference)
 * 
 * RÉFÉRENCES:
 * ----------
 * - Hébert A., "Applied Reactor Physics", Presses Inter. Polytechnique (2008)
 * - Raviart P.A., Thomas J.M., RAIRO Anal. Numér. 11(1), 1977
 * - Smith K.S., "Nodal Method Storage Reduction", Trans. ANS 44, 265 (1983)
 * - Lewis E.E., Miller W.F., "Computational Methods of Neutron Transport" (1984)
 * 
 */

#include "NeutFEM.hpp"
#include <iomanip>
#include <chrono>
#include <fstream>

// ============================================================================
// STRUCTURES DE DONNÉES POUR LES OPTIMISATIONS (définies dans NeutFEM.hpp)
// ============================================================================
// Les structures DiagonalSchurCache et CMFDData sont définies dans NeutFEM.hpp

// ============================================================================
// CONSTRUCTEURS ET INITIALISATION
// ============================================================================
// 
// Architecture mémoire:
// - Les données matériaux sont stockées de façon contiguë par groupe:
//   D_data_[g * n_elem + e] = coefficient de diffusion du groupe g, élément e
// - Les solutions sont également organisées par groupe:
//   Sol_Phi_[g * n_Phi + i] = flux scalaire du groupe g, DOF i
// - Les matrices sont créées une fois et réutilisées à chaque itération
//
// ============================================================================

/**
 * @brief Constructeur avec ordre unique RTₖ-Pₖ (k = order)
 * 
 * Ce constructeur délègue au constructeur à deux ordres avec rt_order = p_order.
 * C'est la configuration la plus courante et la plus stable.
 * 
 * @param order  Ordre polynomial (0, 1, ou 2)
 * @param ng     Nombre de groupes d'énergie
 * @param x_breaks  Coordonnées des nœuds du maillage en X
 * @param y_breaks  Coordonnées des nœuds du maillage en Y (trivial en 1D)
 * @param z_breaks  Coordonnées des nœuds du maillage en Z (trivial en 1D/2D)
 */
NeutFEM::NeutFEM(int order, int ng,
                 const Vec_t& x_breaks,
                 const Vec_t& y_breaks,
                 const Vec_t& z_breaks)
    : NeutFEM(order, order, ng, x_breaks, y_breaks, z_breaks)
{
    // Délègue au constructeur avec ordres distincts (pattern de délégation C++11)
}

/**
 * @brief Constructeur principal avec ordres distincts RTₖ-Pₘ
 * 
 * Ce constructeur permet de spécifier des ordres différents pour le courant (RT)
 * et le flux (P). La condition de stabilité inf-sup requiert k_RT ≥ k_P.
 * 
 * COMBINAISONS VALIDES:
 * - RT0-P0: Volumes finis classiques (courant constant par face)
 * - RT1-P0: Amélioration du courant avec flux constant
 * - RT1-P1: Éléments linéaires complets
 * - RT2-P2: Éléments quadratiques (haute précision)
 * 
 * @warning Si k_RT < k_P, le système est instable (violation inf-sup).
 *          Le constructeur force alors k_P = k_RT avec un avertissement.
 * 
 * @param rt_order  Ordre des éléments de Raviart-Thomas (0, 1, ou 2)
 * @param p_order   Ordre des polynômes de Legendre (0, 1, ou 2)
 * @param ng        Nombre de groupes d'énergie
 * @param x_breaks  Positions des interfaces de mailles en X
 * @param y_breaks  Positions des interfaces de mailles en Y
 * @param z_breaks  Positions des interfaces de mailles en Z
 */
NeutFEM::NeutFEM(int rt_order, int p_order, int ng,
                 const Vec_t& x_breaks,
                 const Vec_t& y_breaks,
                 const Vec_t& z_breaks)
    : mesh_(x_breaks, y_breaks, z_breaks)          // Maillage cartésien structuré
    , fespace_(mesh_,                               // Espace d'éléments finis
               static_cast<RTOrder>(std::min(rt_order, 2)), 
               static_cast<FEOrder>(std::min(p_order, 2)))
    , num_groups_(ng)                               // Nombre de groupes d'énergie
    , rt_order_int_(std::min(rt_order, 2))         // Ordre RT borné à [0,2]
    , p_order_int_(std::min(p_order, 2))           // Ordre P borné à [0,2]
    , rt_order_(static_cast<RTOrder>(std::min(rt_order, 2)))
    , fe_order_(static_cast<FEOrder>(std::min(p_order, 2)))
    , linear_solver_type_(LinearSolverType::BICGSTAB)  // Solveur itératif par défaut
    , has_quarter_symmetry_(false)                 // Symétrie 1/4 de cœur désactivée
    , has_central_symmetry_(false)                 // Symétrie centrale désactivée
    , sym_axis1_(0)                                // Premier axe de symétrie
    , sym_axis2_(1)                                // Second axe de symétrie
    , tol_keff_(1e-5)                              // Convergence sur k-eff
    , tol_flux_(1e-5)                              // Convergence sur le flux
    , tol_L2_(1e-5)                                // Norme L² pour la convergence
    , max_outer_iter_(200)                         // Itérations externes max
    , max_inner_iter_(1000)                        // Itérations internes max
    , verbosity_(VerbosityLevel::NORMAL)           // Niveau de verbosité
    , last_keff_direct_(1.0)                       // Dernier k-eff direct calculé
    , last_keff_adjoint_(1.0)                      // Dernier k-eff adjoint calculé
    , has_valid_keff_(false)                       // Indicateur de validité k-eff
    , has_valid_adjoint_(false)                    // Indicateur de validité adjoint
{
    // =========================================================================
    // VÉRIFICATION DE LA CONDITION DE STABILITÉ INF-SUP
    // =========================================================================
    // Pour que le problème mixte soit bien posé, l'espace RT doit être
    // "assez riche" par rapport à l'espace P. Mathématiquement:
    //   ∃β > 0 : sup_{ψ∈RT_k} <∇·ψ, v> / ||ψ|| ≥ β ||v||  ∀v∈P_m
    // Cette condition est satisfaite si k ≥ m.
    if (rt_order_int_ < p_order_int_) {
        Log(VerbosityLevel::NORMAL, "");
        Log(VerbosityLevel::NORMAL, "!!! ERREUR: RT", rt_order_int_, "-P", p_order_int_, " est instable !!!");
        Log(VerbosityLevel::NORMAL, "    Pour la stabilite inf-sup, il faut k_RT >= k_P");
        Log(VerbosityLevel::NORMAL, "    Combinaisons valides: RT0-P0, RT1-P0, RT1-P1, RT2-P0, RT2-P1, RT2-P2");
        Log(VerbosityLevel::NORMAL, "    Forçage à RT", rt_order_int_, "-P", rt_order_int_);
        Log(VerbosityLevel::NORMAL, "");
        
        // Forcer p_order = rt_order pour éviter l'instabilité
        p_order_int_ = rt_order_int_;
        fe_order_ = static_cast<FEOrder>(rt_order_int_);
        
        // TECHNIQUE AVANCÉE: Reconstruction in-place de fespace_
        // Le membre fespace_ a déjà été construit avec les mauvais ordres.
        // Comme il contient des références constantes, on ne peut pas utiliser
        // l'opérateur d'affectation. On utilise donc la technique de placement new:
        // 1. Appeler explicitement le destructeur
        // 2. Reconstruire l'objet à la même adresse avec les bons paramètres
        fespace_.~FESpace();
        new (&fespace_) FESpace(mesh_, rt_order_, fe_order_);
    }
    
    // =========================================================================
    // ALLOCATION DES TABLEAUX DE DONNÉES
    // =========================================================================
    // Organisation mémoire: tous les tableaux sont de dimension (ng * n_elem)
    // avec un accès linéarisé: data[g * n_elem + e] pour groupe g, élément e.
    // Cette organisation favorise la localité des données lors du balayage
    // par groupe d'énergie.
    
    const int n_elem = mesh_.GetNE();    // Nombre total d'éléments
    const int n_Phi = fespace_.n_Phi;    // DOFs flux par groupe
    const int n_J = fespace_.n_J;        // DOFs courant par groupe
    
    // --- Coefficients de diffusion D_g(r) [cm] ---
    D_data_.resize(ng * n_elem);
    D_data_.setOnes();                   // Valeur par défaut: D = 1 cm
    
    // --- Source externe Q_g(r) [n/cm³/s] ---
    SRC_data_.resize(ng * n_elem);
    SRC_data_.setZero();                 // Pas de source externe par défaut
    
    // --- Section efficace de disparition Σ_r,g(r) [cm⁻¹] ---
    SigR_data_.resize(ng * n_elem);
    SigR_data_.setConstant(0.01);        // Valeur typique pour modérateur
    
    // --- Taux de production de fission νΣ_f,g(r) [cm⁻¹] ---
    NSF_data_.resize(ng * n_elem);
    NSF_data_.setZero();                 // Pas de fission par défaut
    
    // --- Taux de puissance κΣ_f,g(r) [J·cm⁻¹] ---
    KSF_data_.resize(ng * n_elem);
    KSF_data_.setZero();                 // Pas de puissance par défaut
    
    // --- Spectre de fission χ_g(r) [-] ---
    Chi_data_.resize(ng * n_elem);
    Chi_data_.setZero();
    
    // Par défaut, tous les neutrons de fission naissent dans le groupe 1 (rapide)
    if (ng > 0) {
        for (int e = 0; e < n_elem; ++e) {
            Chi_data_(e) = 1.0;  // χ₁ = 1.0, χ_{g>1} = 0.0
        }
    }
    
    // --- Matrice de scattering Σ_{s,g'→g}(r) [cm⁻¹] ---
    // Organisation: SigS_data_[(g * ng + g') * n_elem + e] = Σ_{s,g'→g}(e)
    // Note: g' = groupe source, g = groupe cible
    SigS_data_.resize(ng * ng * n_elem);
    SigS_data_.setZero();
    
    // =========================================================================
    // ALLOCATION DES VECTEURS SOLUTIONS
    // =========================================================================
    // Les solutions sont stockées par groupe de façon contiguë.
    // Sol_Phi_[g * n_Phi + i] = φ_g au DOF i
    
    Sol_Phi_.resize(ng * n_Phi);
    Sol_Phi_.setConstant(1.0);           // Initialisation à flux plat
    Sol_J_.resize(ng * n_J);
    Sol_J_.setZero();                    // Courant nul initialement
    
    // Solutions adjointes (pour calculs de sensibilité)
    Sol_Phi_adj_.resize(ng * n_Phi);
    Sol_Phi_adj_.setConstant(1.0);
    Sol_J_adj_.resize(ng * n_J);
    Sol_J_adj_.setZero();
    
    // =========================================================================
    // ALLOCATION DES MATRICES GLOBALES
    // =========================================================================
    // Matrices du système mixte:
    //   [A   B^T] [J ]   [0]
    //   [B   -C ] [φ] = [S]
    // où A = matrice de masse RT (1/D), B = matrice de divergence, C = masse P (Σ_r)
    
    A_mats_.resize(ng);                  // Une matrice A par groupe (dépend de D_g)
    C_mats_.resize(ng);                  // Une matrice C par groupe (dépend de Σ_r,g)
    M_fiss_.resize(ng);                  // Matrices de production de fission
    M_scatter_.resize(ng * ng);          // Matrices de transfert (scattering)
    M_chi_.resize(ng);                   // Matrices du spectre de fission
    M_nsf_mass_.resize(ng);              // Matrices de νΣ_f pondérées
    
    // =========================================================================
    // INITIALISATION DU SOLVEUR
    // =========================================================================
    
    // Solveur par complément de Schur: S = C + B A⁻¹ Bᵀ
    schur_solver_ = std::make_unique<SchurSolver>();
    
    // =========================================================================
    // INITIALISATION DES CACHES D'OPTIMISATION
    // =========================================================================
    
    // Cache pour solveur diagonal RT0-P0
    diag_schur_cache_ = std::make_unique<DiagonalSchurCache>();
    
    // Données CMFD (initialisées à la demande)
    cmfd_data_ = std::make_unique<CMFDData>();
    
    // Cache de factorisation Schur par groupe (solveurs itératifs)
    // Initialisé lors du premier appel à FactorizeSchurSolvers()
    schur_factorized_ = false;
    
    // Matrices locales pour l'assemblage par élément
    // Ordre de quadrature choisi pour intégrer exactement les produits de 
    // fonctions de forme: 2*max(k_RT, k_P) + 3 points de Gauss
    int quad_order = 2 * std::max(rt_order_int_, p_order_int_) + 3;
    local_matrices_ = std::make_unique<LocalMatrices>(fespace_, quad_order);
    
    // =========================================================================
    // AFFICHAGE DE LA CONFIGURATION
    // =========================================================================
    std::string order_str = "RT" + std::to_string(rt_order_int_) + "-P" + std::to_string(p_order_int_);
    
    Log(VerbosityLevel::NORMAL, "========================================");
    Log(VerbosityLevel::NORMAL, "  NeutFEM - Solveur ", order_str, " (Eigen)");
    Log(VerbosityLevel::NORMAL, "========================================");
    Log(VerbosityLevel::NORMAL, "  Dimension     : ", mesh_.dim, "D");
    Log(VerbosityLevel::NORMAL, "  Maillage      : ", mesh_.nx, " x ", mesh_.ny, " x ", mesh_.nz);
    Log(VerbosityLevel::NORMAL, "  Elements      : ", n_elem);
    Log(VerbosityLevel::NORMAL, "  Groupes       : ", ng);
    Log(VerbosityLevel::NORMAL, "  DOFs flux     : ", n_Phi, " par groupe");
    Log(VerbosityLevel::NORMAL, "  DOFs courant  : ", n_J, " par groupe");
    
    // Afficher les détails des DOFs pour RT
    int n_face_total = fespace_.n_Jx + fespace_.n_Jy + fespace_.n_Jz;
    int n_interior = n_J - n_face_total;
    Log(VerbosityLevel::NORMAL, "    - Face DOFs   : ", n_face_total);
    Log(VerbosityLevel::NORMAL, "    - Interior DOFs: ", n_interior);
    Log(VerbosityLevel::NORMAL, "========================================\n");
}

// ============================================================================
// CONFIGURATION
// ============================================================================

std::string NeutFEM::GetSolverName() const {
    switch (linear_solver_type_) {
        case LinearSolverType::DIRECT_LU:     return "SparseLU";
        case LinearSolverType::DIRECT_LDLT:   return "SimplicialLDLT";
        case LinearSolverType::DIRECT_LLT:    return "SimplicialLLT";
        case LinearSolverType::CG:            return "CG";
        case LinearSolverType::CG_DIAG:       return "CG + Diag";
        case LinearSolverType::CG_ICHOL:      return "CG + IChol";
        case LinearSolverType::BICGSTAB:      return "BiCGSTAB";
        case LinearSolverType::BICGSTAB_DIAG: return "BiCGSTAB + Diag";
        case LinearSolverType::BICGSTAB_ILU:  return "BiCGSTAB + ILU";
        case LinearSolverType::LCG:           return "LSCG";
        default:                              return "Unknown";
    }
}

void NeutFEM::SetLinearSolver(LinearSolverType type) {
    linear_solver_type_ = type;
    schur_solver_->SetSolverType(type);
}

void NeutFEM::SetTolerance(double tol_keff, double tol_flux, double tol_L2,
                           int max_outer, int max_inner) {
    tol_keff_ = tol_keff;
    tol_flux_ = tol_flux;
    tol_L2_ = tol_L2;
    max_outer_iter_ = max_outer;
    max_inner_iter_ = max_inner;
    schur_solver_->SetTolerance(tol_flux_, max_inner);
}

void NeutFEM::SetBC(int attr, BCType type, double value) {
    bc_types_[attr] = type;
    bc_values_[attr] = value;
}

void NeutFEM::SetRobinCoefficients(int attr, double alpha, double beta) {
    robin_alpha_[attr] = alpha;
    robin_beta_[attr] = beta;
}

void NeutFEM::ResetFlux() {
    Sol_Phi_.setConstant(1.0);
    Sol_J_.setZero();
    Sol_Phi_adj_.setConstant(1.0);
    Sol_J_adj_.setZero();
    has_valid_keff_ = false;
    has_valid_adjoint_ = false;
}

void NeutFEM::ApplyQuarterRotationalSymmetry(int axis1, int axis2) {
    has_quarter_symmetry_ = true;
    sym_axis1_ = axis1;
    sym_axis2_ = axis2;
    SetBC(static_cast<int>(BoundaryID::LEFT_2D), BCType::MIRROR, 0.0);
    SetBC(static_cast<int>(BoundaryID::BOTTOM_2D), BCType::MIRROR, 0.0);
}

void NeutFEM::ApplyCentralSymmetry(int axis1, int axis2) {
    has_central_symmetry_ = true;
    sym_axis1_ = axis1;
    sym_axis2_ = axis2;
}

// ============================================================================
// ASSEMBLAGE DES MATRICES GLOBALES
// ============================================================================
// 
// SYSTÈME MIXTE-DUAL:
// ------------------
// Le système variationnel discrétisé s'écrit sous forme matricielle:
//   [A    B^T] [J ]   [g ]
//   [B   -C  ] [φ] = [-S]
// 
// où:
// - A = matrice de masse RT : A[i,j] = ∫ (1/D) ψᵢ · ψⱼ dV
// - B = matrice de divergence : B[i,j] = ∫ φᵢ ∇·ψⱼ dV  
// - C = matrice de réaction : C[i,j] = ∫ Σᵣ φᵢ φⱼ dV
// - g = contribution des conditions aux limites (Robin/Dirichlet)
// - S = source (fission + scattering + externe)
// 
// STRATÉGIE D'ASSEMBLAGE:
// ----------------------
// 1. B est indépendante des données matériaux → assemblée une seule fois
// 2. A et C dépendent du groupe → assemblées pour chaque groupe
// 3. Les conditions aux limites modifient A (termes de surface)
// 
// ============================================================================

/**
 * @brief Assemble toutes les matrices globales du système
 * 
 * OPTIMISATIONS APPLIQUÉES:
 * 1. Pré-allocation optimisée des triplets
 * 2. Skip des coefficients nuls pour le scattering
 */
void NeutFEM::BuildMatrices() {
    Log(VerbosityLevel::NORMAL, "Assemblage des matrices...");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // B est indépendante du groupe - assemblée une seule fois
    AssembleB();
    
    // Pré-calculer les indices non-nuls de scattering
    // pour éviter les tests répétés dans les itérations
    std::vector<std::vector<int>> nonzero_scatter(num_groups_);
    
    // Assemblage par groupe
    for (int g = 0; g < num_groups_; ++g) {
        AssembleA(g);
        ApplyDirichletToA(g);
        AssembleC(g);
        AssembleFissionMatrix(g);
        
        // Assemblage des matrices de scattering avec détection des non-nuls
        for (int gp = 0; gp < num_groups_; ++gp) {
            AssembleScatteringMatrix(gp, g);
            // Enregistrer les indices non-nuls pour optimisation ultérieure
            const int idx = g * num_groups_ + gp;
            if (M_scatter_[idx].nonZeros() > 0 && gp != g) {
                nonzero_scatter[g].push_back(gp);
            }
        }
        
        // Assemblage des matrices pondérées χ et νΣf
        {
            Vec_t chi_coeff(mesh_.GetNE());
            const int n_elem = mesh_.GetNE();
            // Accès direct sans appel de fonction
            for (int e = 0; e < n_elem; ++e)
                chi_coeff(e) = Chi_data_(g * n_elem + e);
            M_chi_[g] = AssembleWeightedMassMatrix(chi_coeff);
        }
        {
            Vec_t nsf_coeff(mesh_.GetNE());
            const int n_elem = mesh_.GetNE();
            for (int e = 0; e < n_elem; ++e)
                nsf_coeff(e) = NSF_data_(g * n_elem + e);
            M_nsf_mass_[g] = AssembleWeightedMassMatrix(nsf_coeff);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    Log(VerbosityLevel::NORMAL, "  Assemblage termine en ", elapsed, " ms");
    
    // Invalider les caches (les matrices ont changé)
    schur_factorized_ = false;
    if (diag_schur_cache_) diag_schur_cache_->is_valid = false;
    if (cmfd_data_) cmfd_data_->is_initialized = false;
}

// ============================================================================
// SOLVEUR DIAGONAL RT0-P0 (OPTIMISATION MAJEURE)
// ============================================================================
//
// Pour RT0-P0, le complément de Schur S = C + B A⁻¹ Bᵀ est DIAGONAL.
// Ceci est dû au fait que:
// - Chaque élément a 1 DOF flux (P0) et 2*dim DOFs courant (faces)
// - A est diagonale par blocs (couplage intra-élément uniquement)
// - B connecte chaque élément uniquement à ses propres faces
//
// Résultat: S(e,e) = C(e,e) + Σ_f [B(e,f)]² / A(f,f)
//
// L'inversion est triviale: φ = S⁻¹ × rhs (opération O(n))
//
// ============================================================================

/**
 * @brief Construit le cache du solveur diagonal pour RT0-P0
 * 
 * Pré-calcule 1/S_ii pour chaque groupe, permettant une résolution
 * instantanée du système Schur.
 * 
 * @note Cette méthode ne fait rien si rt_order > 0 ou p_order > 0
 */
void NeutFEM::BuildDiagonalSchurCache() {
    // Vérifier que nous sommes en RT0-P0
    if (rt_order_int_ != 0 || p_order_int_ != 0) {
        Log(VerbosityLevel::NORMAL, "  Cache diagonal: non applicable (ordre > 0)");
        return;
    }
    
    if (diag_schur_cache_->is_valid) {
        return;  // Déjà construit
    }
    
    // Vérifier que les matrices sont assemblées
    if (A_mats_.empty() || A_mats_[0].rows() == 0) {
        Log(VerbosityLevel::NORMAL, "  Cache diagonal: matrices non assemblées, skip");
        return;
    }
    
    Log(VerbosityLevel::NORMAL, "  Construction du cache diagonal RT0-P0...");
    auto start = std::chrono::high_resolution_clock::now();
    
    const int ng = num_groups_;
    const int n_elem = mesh_.GetNE();
    const int dim = mesh_.dim;
    const int n_J = fespace_.n_J;
    
    diag_schur_cache_->S_diag_inv.resize(ng);
    
    for (int g = 0; g < ng; ++g) {
        Vec& S_inv = diag_schur_cache_->S_diag_inv[g];
        S_inv.resize(n_elem);
        
        // Vérifier les dimensions des matrices
        if (A_mats_[g].rows() != n_J || C_mats_[g].rows() != n_elem) {
            Log(VerbosityLevel::NORMAL, "  Cache diagonal: dimensions incorrectes pour groupe ", g);
            S_inv.setOnes();  // Fallback
            continue;
        }
        
        // Pour chaque élément, calculer S(e,e) = C(e,e) + Σ_f [B(e,f)]² / A(f,f)
        for (int iz = 0; iz < mesh_.nz; ++iz) {
            for (int iy = 0; iy < mesh_.ny; ++iy) {
                for (int ix = 0; ix < mesh_.nx; ++ix) {
                    const int e = mesh_.ElemIndex(ix, iy, iz);
                    
                    // Terme C(e,e) = Σr × Volume
                    double S_ee = C_mats_[g].coeff(e, e);
                    
                    // Contribution des faces en X
                    {
                        int f_left = fespace_.JxFaceIndex(ix, iy, iz);
                        int f_right = fespace_.JxFaceIndex(ix + 1, iy, iz);
                        
                        // Vérification des bornes
                        if (f_left >= 0 && f_left < n_J && f_right >= 0 && f_right < n_J) {
                            double B_left = B_mat_.coeff(e, f_left);
                            double B_right = B_mat_.coeff(e, f_right);
                            double A_left = A_mats_[g].coeff(f_left, f_left);
                            double A_right = A_mats_[g].coeff(f_right, f_right);
                            
                            if (std::abs(A_left) > 1e-14)
                                S_ee += B_left * B_left / A_left;
                            if (std::abs(A_right) > 1e-14)
                                S_ee += B_right * B_right / A_right;
                        }
                    }
                    
                    // Contribution des faces en Y (2D et 3D)
                    if (dim >= 2) {
                        int f_bottom = fespace_.JyFaceIndex(ix, iy, iz);
                        int f_top = fespace_.JyFaceIndex(ix, iy + 1, iz);
                        
                        if (f_bottom >= 0 && f_bottom < n_J && f_top >= 0 && f_top < n_J) {
                            double B_bottom = B_mat_.coeff(e, f_bottom);
                            double B_top = B_mat_.coeff(e, f_top);
                            double A_bottom = A_mats_[g].coeff(f_bottom, f_bottom);
                            double A_top = A_mats_[g].coeff(f_top, f_top);
                            
                            if (std::abs(A_bottom) > 1e-14)
                                S_ee += B_bottom * B_bottom / A_bottom;
                            if (std::abs(A_top) > 1e-14)
                                S_ee += B_top * B_top / A_top;
                        }
                    }
                    
                    // Contribution des faces en Z (3D)
                    if (dim >= 3) {
                        int f_back = fespace_.JzFaceIndex(ix, iy, iz);
                        int f_front = fespace_.JzFaceIndex(ix, iy, iz + 1);
                        
                        if (f_back >= 0 && f_back < n_J && f_front >= 0 && f_front < n_J) {
                            double B_back = B_mat_.coeff(e, f_back);
                            double B_front = B_mat_.coeff(e, f_front);
                            double A_back = A_mats_[g].coeff(f_back, f_back);
                            double A_front = A_mats_[g].coeff(f_front, f_front);
                            
                            if (std::abs(A_back) > 1e-14)
                                S_ee += B_back * B_back / A_back;
                            if (std::abs(A_front) > 1e-14)
                                S_ee += B_front * B_front / A_front;
                        }
                    }
                    
                    // Stocker l'inverse
                    S_inv(e) = (std::abs(S_ee) > 1e-14) ? 1.0 / S_ee : 0.0;
                }
            }
        }
    }
    
    diag_schur_cache_->is_valid = true;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start).count();
    Log(VerbosityLevel::NORMAL, "    Cache diagonal construit en ", elapsed, " ms");
}

/**
 * @brief Résout le système Schur en utilisant le cache diagonal (RT0-P0)
 * 
 * @param g       Groupe d'énergie
 * @param rhs     Second membre
 * @param Phi_sol Solution flux (sortie)
 * @param J_sol   Solution courant (sortie)
 */
void NeutFEM::SolveDiagonalSchur(int g, const Vec& rhs, Vec& Phi_sol, Vec& J_sol) {
    const int n_elem = mesh_.GetNE();
    const int n_J = fespace_.n_J;
    
    // Résolution triviale: φ = S⁻¹ × rhs
    const Vec& S_inv = diag_schur_cache_->S_diag_inv[g];
    Phi_sol = S_inv.array() * rhs.array();
    
    // Reconstruction du courant: J = A⁻¹ × Bᵀ × φ
    // Pour RT0, A est diagonale donc A⁻¹ est trivial
    J_sol.resize(n_J);
    J_sol.setZero();
    
    // J_f = (1/A_ff) × Σ_e B(e,f) × φ_e
    // Pour RT0, chaque face connecte au plus 2 éléments
    for (int f = 0; f < n_J; ++f) {
        double A_ff = A_mats_[g].coeff(f, f);
        if (std::abs(A_ff) < 1e-14) continue;
        
        double sum = 0.0;
        // Parcourir les éléments connectés à cette face
        for (SpMat::InnerIterator it(BT_mat_, f); it; ++it) {
            int e = it.row();
            sum += it.value() * Phi_sol(e);
        }
        J_sol(f) = sum / A_ff;
    }
}

// ============================================================================
// ACCÉLÉRATION CMFD (Coarse Mesh Finite Difference)
// ============================================================================
//
// Le CMFD est une méthode d'accélération non-linéaire qui:
// 1. Résout un système volumes-finis grossier (même maillage, schéma simplifié)
// 2. Utilise les courants fins pour corriger les coefficients de diffusion
// 3. Applique la correction au flux fin
//
// Avantages:
// - Convergence beaucoup plus rapide que Chebyshev pour les grands systèmes
// - Robuste pour les systèmes faiblement couplés
// - Coût par itération très faible
//
// Algorithme (par itération externe):
// 1. Calculer les courants nets aux faces: J_net,f = J_f (du calcul fin)
// 2. Calculer D̃_f = D_f × A_f / Δx (coefficient de fuite standard)
// 3. Calculer D̂_f = J_net,f / (φ_L - φ_R) - D̃_f (correction non-linéaire)
// 4. Assembler et résoudre le système CMFD: M_cmfd × φ_cmfd = S_cmfd
// 5. Appliquer la correction: φ_new = φ_old × (φ_cmfd / φ_cmfd_old)
//
// ============================================================================

/**
 * @brief Initialise les structures de données CMFD
 */
void NeutFEM::InitializeCMFD() {
    if (cmfd_data_->is_initialized) return;
    
    Log(VerbosityLevel::NORMAL, "  Initialisation CMFD...");
    
    const int ng = num_groups_;
    const int nx = mesh_.nx;
    const int ny = mesh_.ny;
    const int nz = mesh_.nz;
    
    // Allocation des coefficients de couplage
    cmfd_data_->Dtilde_x.resize(ng);
    cmfd_data_->Dtilde_y.resize(ng);
    cmfd_data_->Dtilde_z.resize(ng);
    cmfd_data_->Dhat_x.resize(ng);
    cmfd_data_->Dhat_y.resize(ng);
    cmfd_data_->Dhat_z.resize(ng);
    cmfd_data_->M_cmfd.resize(ng);
    
    const int n_faces_x = (nx + 1) * ny * nz;
    const int n_faces_y = nx * (ny + 1) * nz;
    const int n_faces_z = nx * ny * (nz + 1);
    
    for (int g = 0; g < ng; ++g) {
        cmfd_data_->Dtilde_x[g].resize(n_faces_x);
        cmfd_data_->Dtilde_y[g].resize(n_faces_y);
        cmfd_data_->Dtilde_z[g].resize(n_faces_z);
        cmfd_data_->Dhat_x[g].resize(n_faces_x);
        cmfd_data_->Dhat_y[g].resize(n_faces_y);
        cmfd_data_->Dhat_z[g].resize(n_faces_z);
        
        cmfd_data_->Dtilde_x[g].setZero();
        cmfd_data_->Dtilde_y[g].setZero();
        cmfd_data_->Dtilde_z[g].setZero();
        cmfd_data_->Dhat_x[g].setZero();
        cmfd_data_->Dhat_y[g].setZero();
        cmfd_data_->Dhat_z[g].setZero();
    }
    
    // Pré-calculer D̃ (coefficients de diffusion standard)
    ComputeDtildeCoefficients();
    
    cmfd_data_->is_initialized = true;
    Log(VerbosityLevel::NORMAL, "    CMFD initialise");
}

/**
 * @brief Calcule les coefficients D̃ (diffusion standard aux faces)
 * 
 * D̃_f = 2 × D_L × D_R / (D_L × Δx_R + D_R × Δx_L)
 * (moyenne harmonique pondérée par les distances)
 */
void NeutFEM::ComputeDtildeCoefficients() {
    const int ng = num_groups_;
    const int nx = mesh_.nx;
    const int ny = mesh_.ny;
    const int nz = mesh_.nz;
    const int n_elem = mesh_.GetNE();
    
    for (int g = 0; g < ng; ++g) {
        // Faces X
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix <= nx; ++ix) {
                    int f_idx = iz * (ny * (nx + 1)) + iy * (nx + 1) + ix;
                    
                    if (ix == 0 || ix == nx) {
                        // Face frontière - utiliser condition aux limites
                        int e = (ix == 0) ? mesh_.ElemIndex(0, iy, iz) 
                                          : mesh_.ElemIndex(nx - 1, iy, iz);
                        double D = D_data_(g * n_elem + e);
                        double dx = mesh_.x_breaks(ix == 0 ? 1 : nx) - mesh_.x_breaks(ix == 0 ? 0 : nx - 1);
                        cmfd_data_->Dtilde_x[g](f_idx) = 2.0 * D / dx;
                    } else {
                        // Face interne - moyenne harmonique
                        int e_L = mesh_.ElemIndex(ix - 1, iy, iz);
                        int e_R = mesh_.ElemIndex(ix, iy, iz);
                        double D_L = D_data_(g * n_elem + e_L);
                        double D_R = D_data_(g * n_elem + e_R);
                        double dx_L = mesh_.x_breaks(ix) - mesh_.x_breaks(ix - 1);
                        double dx_R = mesh_.x_breaks(ix + 1) - mesh_.x_breaks(ix);
                        
                        cmfd_data_->Dtilde_x[g](f_idx) = 2.0 * D_L * D_R / 
                            (D_L * dx_R + D_R * dx_L);
                    }
                }
            }
        }
        
        // Faces Y (2D et 3D)
        if (mesh_.dim >= 2) {
            for (int iz = 0; iz < nz; ++iz) {
                for (int iy = 0; iy <= ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        int f_idx = iz * ((ny + 1) * nx) + iy * nx + ix;
                        
                        if (iy == 0 || iy == ny) {
                            int e = (iy == 0) ? mesh_.ElemIndex(ix, 0, iz)
                                              : mesh_.ElemIndex(ix, ny - 1, iz);
                            double D = D_data_(g * n_elem + e);
                            double dy = mesh_.y_breaks(iy == 0 ? 1 : ny) - mesh_.y_breaks(iy == 0 ? 0 : ny - 1);
                            cmfd_data_->Dtilde_y[g](f_idx) = 2.0 * D / dy;
                        } else {
                            int e_L = mesh_.ElemIndex(ix, iy - 1, iz);
                            int e_R = mesh_.ElemIndex(ix, iy, iz);
                            double D_L = D_data_(g * n_elem + e_L);
                            double D_R = D_data_(g * n_elem + e_R);
                            double dy_L = mesh_.y_breaks(iy) - mesh_.y_breaks(iy - 1);
                            double dy_R = mesh_.y_breaks(iy + 1) - mesh_.y_breaks(iy);
                            
                            cmfd_data_->Dtilde_y[g](f_idx) = 2.0 * D_L * D_R /
                                (D_L * dy_R + D_R * dy_L);
                        }
                    }
                }
            }
        }
        
        // Faces Z (3D)
        if (mesh_.dim >= 3) {
            for (int iz = 0; iz <= nz; ++iz) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        int f_idx = iz * (ny * nx) + iy * nx + ix;
                        
                        if (iz == 0 || iz == nz) {
                            int e = (iz == 0) ? mesh_.ElemIndex(ix, iy, 0)
                                              : mesh_.ElemIndex(ix, iy, nz - 1);
                            double D = D_data_(g * n_elem + e);
                            double dz = mesh_.z_breaks(iz == 0 ? 1 : nz) - mesh_.z_breaks(iz == 0 ? 0 : nz - 1);
                            cmfd_data_->Dtilde_z[g](f_idx) = 2.0 * D / dz;
                        } else {
                            int e_L = mesh_.ElemIndex(ix, iy, iz - 1);
                            int e_R = mesh_.ElemIndex(ix, iy, iz);
                            double D_L = D_data_(g * n_elem + e_L);
                            double D_R = D_data_(g * n_elem + e_R);
                            double dz_L = mesh_.z_breaks(iz) - mesh_.z_breaks(iz - 1);
                            double dz_R = mesh_.z_breaks(iz + 1) - mesh_.z_breaks(iz);
                            
                            cmfd_data_->Dtilde_z[g](f_idx) = 2.0 * D_L * D_R /
                                (D_L * dz_R + D_R * dz_L);
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Met à jour les facteurs de correction D̂ basés sur les courants fins
 * 
 * D̂_f = J_f / (φ_L - φ_R) - D̃_f
 * où J_f est le courant net à la face f calculé par le solveur fin.
 */
void NeutFEM::UpdateDhatCoefficients() {
    const int ng = num_groups_;
    const int nx = mesh_.nx;
    const int ny = mesh_.ny;
    const int nz = mesh_.nz;
    const int n_Phi = fespace_.n_Phi;
    const int n_J = fespace_.n_J;
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    
    for (int g = 0; g < ng; ++g) {
        // Accès au flux et courant du groupe g
        Eigen::Map<const Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
        Eigen::Map<const Vec> J_g(Sol_J_.data() + g * n_J, n_J);
        
        // Faces X
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix <= nx; ++ix) {
                    int f_idx = iz * (ny * (nx + 1)) + iy * (nx + 1) + ix;
                    int j_idx = fespace_.JxFaceIndex(ix, iy, iz);
                    
                    double J_net = J_g(j_idx);
                    double phi_diff;
                    
                    if (ix == 0) {
                        // Face gauche: φ_L = 0 (vacuum) ou condition spéciale
                        int e_R = mesh_.ElemIndex(0, iy, iz);
                        phi_diff = -Phi_g(e_R * dofs_per_elem);  // Convention: J positif vers +x
                    } else if (ix == nx) {
                        // Face droite
                        int e_L = mesh_.ElemIndex(nx - 1, iy, iz);
                        phi_diff = Phi_g(e_L * dofs_per_elem);
                    } else {
                        // Face interne
                        int e_L = mesh_.ElemIndex(ix - 1, iy, iz);
                        int e_R = mesh_.ElemIndex(ix, iy, iz);
                        phi_diff = Phi_g(e_L * dofs_per_elem) - Phi_g(e_R * dofs_per_elem);
                    }
                    
                    // D̂ = J / Δφ - D̃ (avec protection division par zéro)
                    if (std::abs(phi_diff) > 1e-14) {
                        cmfd_data_->Dhat_x[g](f_idx) = J_net / phi_diff - cmfd_data_->Dtilde_x[g](f_idx);
                    } else {
                        cmfd_data_->Dhat_x[g](f_idx) = 0.0;
                    }
                }
            }
        }
        
        // Faces Y et Z (similaire)
        // ... (code similaire pour Y et Z)
    }
}

/**
 * @brief Applique une correction CMFD au flux
 * 
 * @param g           Groupe d'énergie
 * @param total_fiss  Source de fission totale
 * @param keff        Facteur de multiplication actuel
 * @return Correction de flux à appliquer
 */
Vec NeutFEM::ApplyCMFDCorrection(int g, const Vec& total_fiss, double keff) {
    const int n_elem = mesh_.GetNE();
    const int nx = mesh_.nx;
    const int ny = mesh_.ny;
    const int nz = mesh_.nz;
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    const int n_Phi = fespace_.n_Phi;
    
    // Extraire le flux P0 (moyenne par maille)
    Vec phi_coarse(n_elem);
    Eigen::Map<const Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
    for (int e = 0; e < n_elem; ++e) {
        phi_coarse(e) = Phi_g(e * dofs_per_elem);
    }
    
    // Assembler la matrice CMFD
    // M_cmfd(e,e) = Σr × V + Σ_faces (D̃ + D̂) × A_face
    // M_cmfd(e,e') = -(D̃ + D̂) × A_face (si e et e' sont voisins)
    
    std::vector<Triplet> triplets;
    triplets.reserve(n_elem * 7);  // 1 diag + 6 voisins max
    
    Vec rhs_cmfd(n_elem);
    rhs_cmfd.setZero();
    
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const int e = mesh_.ElemIndex(ix, iy, iz);
                double diag = C_mats_[g].coeff(e * dofs_per_elem, e * dofs_per_elem);
                
                // Contribution des faces X
                {
                    double A_face = mesh_.FaceArea(e, 0);
                    int f_left = iz * (ny * (nx + 1)) + iy * (nx + 1) + ix;
                    int f_right = iz * (ny * (nx + 1)) + iy * (nx + 1) + ix + 1;
                    
                    double Deff_left = cmfd_data_->Dtilde_x[g](f_left) + cmfd_data_->Dhat_x[g](f_left);
                    double Deff_right = cmfd_data_->Dtilde_x[g](f_right) + cmfd_data_->Dhat_x[g](f_right);
                    
                    diag += (Deff_left + Deff_right) * A_face;
                    
                    if (ix > 0) {
                        int e_left = mesh_.ElemIndex(ix - 1, iy, iz);
                        triplets.emplace_back(e, e_left, -Deff_left * A_face);
                    }
                    if (ix < nx - 1) {
                        int e_right = mesh_.ElemIndex(ix + 1, iy, iz);
                        triplets.emplace_back(e, e_right, -Deff_right * A_face);
                    }
                }
                
                // Faces Y (2D/3D)
                if (mesh_.dim >= 2) {
                    double A_face = mesh_.FaceArea(e, 1);
                    int f_bottom = iz * ((ny + 1) * nx) + iy * nx + ix;
                    int f_top = iz * ((ny + 1) * nx) + (iy + 1) * nx + ix;
                    
                    double Deff_bottom = cmfd_data_->Dtilde_y[g](f_bottom) + cmfd_data_->Dhat_y[g](f_bottom);
                    double Deff_top = cmfd_data_->Dtilde_y[g](f_top) + cmfd_data_->Dhat_y[g](f_top);
                    
                    diag += (Deff_bottom + Deff_top) * A_face;
                    
                    if (iy > 0) {
                        int e_bottom = mesh_.ElemIndex(ix, iy - 1, iz);
                        triplets.emplace_back(e, e_bottom, -Deff_bottom * A_face);
                    }
                    if (iy < ny - 1) {
                        int e_top = mesh_.ElemIndex(ix, iy + 1, iz);
                        triplets.emplace_back(e, e_top, -Deff_top * A_face);
                    }
                }
                
                // Faces Z (3D)
                if (mesh_.dim >= 3) {
                    double A_face = mesh_.FaceArea(e, 2);
                    int f_back = iz * (ny * nx) + iy * nx + ix;
                    int f_front = (iz + 1) * (ny * nx) + iy * nx + ix;
                    
                    double Deff_back = cmfd_data_->Dtilde_z[g](f_back) + cmfd_data_->Dhat_z[g](f_back);
                    double Deff_front = cmfd_data_->Dtilde_z[g](f_front) + cmfd_data_->Dhat_z[g](f_front);
                    
                    diag += (Deff_back + Deff_front) * A_face;
                    
                    if (iz > 0) {
                        int e_back = mesh_.ElemIndex(ix, iy, iz - 1);
                        triplets.emplace_back(e, e_back, -Deff_back * A_face);
                    }
                    if (iz < nz - 1) {
                        int e_front = mesh_.ElemIndex(ix, iy, iz + 1);
                        triplets.emplace_back(e, e_front, -Deff_front * A_face);
                    }
                }
                
                triplets.emplace_back(e, e, diag);
                
                // RHS: source de fission + scattering
                double chi_val = Chi_data_(g * n_elem + e);
                rhs_cmfd(e) = chi_val * total_fiss(e * dofs_per_elem) / keff;
            }
        }
    }
    
    // Assembler et résoudre le système CMFD
    SpMat M_cmfd(n_elem, n_elem);
    M_cmfd.setFromTriplets(triplets.begin(), triplets.end());
    
    // Résolution par CG (système SPD)
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    cg.setTolerance(1e-8);
    cg.setMaxIterations(100);
    cg.compute(M_cmfd);
    
    Vec phi_cmfd_new = cg.solve(rhs_cmfd);
    
    // Calculer le facteur de correction: ratio = φ_cmfd_new / φ_coarse
    Vec correction(n_Phi);
    correction.setOnes();
    
    for (int e = 0; e < n_elem; ++e) {
        double ratio = 1.0;
        if (std::abs(phi_coarse(e)) > 1e-14) {
            ratio = phi_cmfd_new(e) / phi_coarse(e);
            // Limiter la correction pour la stabilité
            ratio = std::max(0.5, std::min(2.0, ratio));
        }
        
        // Appliquer à tous les DOFs de l'élément
        for (int d = 0; d < dofs_per_elem; ++d) {
            correction(e * dofs_per_elem + d) = ratio;
        }
    }
    
    // Relaxation
    const double omega = cmfd_data_->relaxation;
    correction = omega * correction + (1.0 - omega) * Vec::Ones(n_Phi);
    
    return correction;
}

/**
 * @brief Assemble la matrice de masse RT : A[i,j] = ∫ (1/D) ψᵢ · ψⱼ dV
 * 
 * Cette matrice encode la loi de Fick J = -D∇φ sous forme faible.
 * Le coefficient 1/D pondère l'intégrale scalaire des fonctions de base RT.
 * 
 * STRUCTURE CREUSE:
 * ----------------
 * Pour RT0: A est bloc-diagonale par élément (couplage par faces uniquement)
 * Pour RT1/RT2: couplages additionnels via les DOFs intérieurs (bulles)
 * 
 * DÉPENDANCE AU GROUPE:
 * --------------------
 * D varie avec le groupe d'énergie → une matrice A par groupe.
 * 
 * @param group  Indice du groupe d'énergie (0 à ng-1)
 */
void NeutFEM::AssembleA(int group) {
    const int n_J = fespace_.n_J;
    std::vector<Triplet> triplets;
    
    // Estimation généreuse du nombre de triplets
    int n_J_local = local_matrices_->NumJDofs();
    triplets.reserve(mesh_.GetNE() * n_J_local * n_J_local);
    
    std::vector<int> J_indices;
    
    for (int iz = 0; iz < mesh_.nz; ++iz) {
        for (int iy = 0; iy < mesh_.ny; ++iy) {
            for (int ix = 0; ix < mesh_.nx; ++ix) {
                const int e = mesh_.ElemIndex(ix, iy, iz);
                const double D = D_data_(group * mesh_.GetNE() + e);
                
		// Calcul de la matrice locale A avec LocalMatrices
                // Note: Sigma = 0 car on veut seulement A, pas C
                local_matrices_->Compute(e, D, 0.0);
                
                // Obtenir les indices globaux pour J (faces + intérieurs)
                local_matrices_->GetGlobalJIndices(ix, iy, iz, J_indices);
                
                const Mat& A_loc = local_matrices_->GetA();
                
                // Assemblage dans la matrice globale
                for (int i = 0; i < A_loc.rows(); ++i) {
                    for (int j = 0; j < A_loc.cols(); ++j) {
                        if (std::abs(A_loc(i, j)) > 1e-14) {
                            triplets.emplace_back(J_indices[i], J_indices[j], A_loc(i, j));
                        }
                    }
                }
            }
        }
    }
    
    A_mats_[group].resize(n_J, n_J);
    A_mats_[group].setFromTriplets(triplets.begin(), triplets.end());
    A_mats_[group].makeCompressed();
}

/**
 * @brief Assemble la matrice de divergence : B[i,j] = ∫ φᵢ ∇·ψⱼ dV
 * 
 * Cette matrice couple le flux scalaire φ et le courant J via l'opérateur
 * de divergence. Elle est la pierre angulaire du schéma mixte.
 * 
 * PROPRIÉTÉS MATHÉMATIQUES:
 * ------------------------
 * - B est rectangulaire: (n_Phi × n_J)
 * - Bᵀ est la matrice gradient discrète
 * - Le noyau de B correspond aux courants à divergence nulle
 * - L'image de Bᵀ est l'espace des flux à moyenne nulle
 * 
 * INDÉPENDANCE DES DONNÉES:
 * ------------------------
 * B ne dépend que de la géométrie et de l'ordre polynomial, pas des
 * sections efficaces. Elle est donc assemblée une seule fois.
 */
void NeutFEM::AssembleB() {
    const int n_Phi = fespace_.n_Phi;
    const int n_J = fespace_.n_J;
    
    std::vector<Triplet> triplets;
    
    int n_J_local = local_matrices_->NumJDofs();
    int n_Phi_local = local_matrices_->NumPhiDofs();
    triplets.reserve(mesh_.GetNE() * n_Phi_local * n_J_local);
    
    std::vector<int> J_indices;
    std::vector<int> Phi_indices;
    
    for (int iz = 0; iz < mesh_.nz; ++iz) {
        for (int iy = 0; iy < mesh_.ny; ++iy) {
            for (int ix = 0; ix < mesh_.nx; ++ix) {
                const int e = mesh_.ElemIndex(ix, iy, iz);
                
                // Calcul de la matrice locale B
                // D = 1.0 et Sigma = 0.0 car seule B est utilisée
                local_matrices_->Compute(e, 1.0, 0.0);
                
                // Obtenir les indices globaux
                local_matrices_->GetGlobalJIndices(ix, iy, iz, J_indices);
                local_matrices_->GetGlobalPhiIndices(ix, iy, iz, Phi_indices);
                
                const Mat& B_loc = local_matrices_->GetB();
                
                // Assemblage: B_loc est (n_Phi_local × n_J_local)
                for (int i = 0; i < B_loc.rows(); ++i) {
                    for (int j = 0; j < B_loc.cols(); ++j) {
                        if (std::abs(B_loc(i, j)) > 1e-14) {
                            triplets.emplace_back(Phi_indices[i], J_indices[j], B_loc(i, j));
                        }
                    }
                }
            }
        }
    }
    
    B_mat_.resize(n_Phi, n_J);
    B_mat_.setFromTriplets(triplets.begin(), triplets.end());
    B_mat_.makeCompressed();
    BT_mat_ = B_mat_.transpose();
}

/**
 * @brief Assemble la matrice de réaction : C[i,j] = ∫ Σᵣ φᵢ φⱼ dV
 * 
 * Cette matrice encode l'absorption et l'auto-diffusion (self-scatter)
 * des neutrons dans chaque groupe d'énergie.
 * 
 * SECTION EFFICACE DE DISPARITION:
 * -------------------------------
 * Σᵣ = Σₐ + Σₛ,out = Σₜ - Σₛ,self
 * où:
 * - Σₐ = absorption (capture + fission)
 * - Σₛ,out = diffusion vers d'autres groupes
 * - Σₛ,self = self-scatter (même groupe)
 * 
 * STRUCTURE:
 * ---------
 * Pour P0: C est diagonale (1 DOF par élément)
 * Pour P1/P2: C a une structure bloc-diagonale (couplage intra-élément)
 * 
 * @param group  Indice du groupe d'énergie
 */
void NeutFEM::AssembleC(int group) {
    const int n_Phi = fespace_.n_Phi;
    std::vector<Triplet> triplets;
    
    int n_Phi_local = local_matrices_->NumPhiDofs();
    triplets.reserve(mesh_.GetNE() * n_Phi_local * n_Phi_local);
    
    std::vector<int> Phi_indices;
    
    for (int iz = 0; iz < mesh_.nz; ++iz) {
        for (int iy = 0; iy < mesh_.ny; ++iy) {
            for (int ix = 0; ix < mesh_.nx; ++ix) {
                const int e = mesh_.ElemIndex(ix, iy, iz);
                const double Sigma = SigR_data_(group * mesh_.GetNE() + e);
                
                // Calcul de la matrice locale C
                // D = 1.0 (pas utilisé pour C)
                local_matrices_->Compute(e, 1.0, Sigma);
                
                // Obtenir les indices globaux
                local_matrices_->GetGlobalPhiIndices(ix, iy, iz, Phi_indices);
                
                const Mat& C_loc = local_matrices_->GetC();
                
                // Assemblage
                for (int i = 0; i < C_loc.rows(); ++i) {
                    for (int j = 0; j < C_loc.cols(); ++j) {
                        if (std::abs(C_loc(i, j)) > 1e-14) {
                            triplets.emplace_back(Phi_indices[i], Phi_indices[j], C_loc(i, j));
                        }
                    }
                }
            }
        }
    }
    
    C_mats_[group].resize(n_Phi, n_Phi);
    C_mats_[group].setFromTriplets(triplets.begin(), triplets.end());
    C_mats_[group].makeCompressed();
}

void NeutFEM::AssembleFissionMatrix(int group) {
    const int n_Phi = fespace_.n_Phi;
    const int n_elem = mesh_.GetNE();
    std::vector<Triplet> triplets;
    
    if (p_order_int_ == 0) {
        // P0 : matrice diagonale simple
        for (int e = 0; e < n_elem; ++e) {
            double nsf = NSF_data_(group * n_elem + e);
            if (std::abs(nsf) > 1e-14) {
                triplets.emplace_back(e, e, nsf * mesh_.ElemVolume(e));
            }
        }
    } else {
        // P1/P2 : utiliser LocalMatrices pour construire la matrice de masse
        // pondérée par νΣf
        std::vector<int> Phi_indices;
        int n_Phi_local = local_matrices_->NumPhiDofs();
        
        for (int iz = 0; iz < mesh_.nz; ++iz) {
            for (int iy = 0; iy < mesh_.ny; ++iy) {
                for (int ix = 0; ix < mesh_.nx; ++ix) {
                    const int e = mesh_.ElemIndex(ix, iy, iz);
                    const double nsf = NSF_data_(group * n_elem + e);
                    
                    if (std::abs(nsf) < 1e-14) continue;
                    
                    // Utiliser la matrice C avec Sigma = nsf
                    local_matrices_->Compute(e, 1.0, nsf);
                    local_matrices_->GetGlobalPhiIndices(ix, iy, iz, Phi_indices);
                    
                    const Mat& C_loc = local_matrices_->GetC();
                    
                    for (int i = 0; i < n_Phi_local; ++i) {
                        for (int j = 0; j < n_Phi_local; ++j) {
                            if (std::abs(C_loc(i, j)) > 1e-14) {
                                triplets.emplace_back(Phi_indices[i], Phi_indices[j], C_loc(i, j));
                            }
                        }
                    }
                }
            }
        }
    }
    
    M_fiss_[group].resize(n_Phi, n_Phi);
    M_fiss_[group].setFromTriplets(triplets.begin(), triplets.end());
    M_fiss_[group].makeCompressed();
}

void NeutFEM::AssembleScatteringMatrix(int g_from, int g_to) {
    const int idx = g_to * num_groups_ + g_from;
    const int n_Phi = fespace_.n_Phi;
    const int n_elem = mesh_.GetNE();
    const int offset = GetSigSOffset(g_from, g_to);
    std::vector<Triplet> triplets;
    
    if (p_order_int_ == 0) {
        // P0 : matrice diagonale simple
        for (int e = 0; e < n_elem; ++e) {
            double sigs = SigS_data_(offset + e);
            if (std::abs(sigs) > 1e-14) {
                triplets.emplace_back(e, e, sigs * mesh_.ElemVolume(e));
            }
        }
    } else {
        // P1/P2 : utiliser LocalMatrices
        std::vector<int> Phi_indices;
        int n_Phi_local = local_matrices_->NumPhiDofs();
        
        for (int iz = 0; iz < mesh_.nz; ++iz) {
            for (int iy = 0; iy < mesh_.ny; ++iy) {
                for (int ix = 0; ix < mesh_.nx; ++ix) {
                    const int e = mesh_.ElemIndex(ix, iy, iz);
                    const double sigs = SigS_data_(offset + e);
                    
                    if (std::abs(sigs) < 1e-14) continue;
                    
                    local_matrices_->Compute(e, 1.0, sigs);
                    local_matrices_->GetGlobalPhiIndices(ix, iy, iz, Phi_indices);
                    
                    const Mat& C_loc = local_matrices_->GetC();
                    
                    for (int i = 0; i < n_Phi_local; ++i) {
                        for (int j = 0; j < n_Phi_local; ++j) {
                            if (std::abs(C_loc(i, j)) > 1e-14) {
                                triplets.emplace_back(Phi_indices[i], Phi_indices[j], C_loc(i, j));
                            }
                        }
                    }
                }
            }
        }
    }
    
    M_scatter_[idx].resize(n_Phi, n_Phi);
    M_scatter_[idx].setFromTriplets(triplets.begin(), triplets.end());
    M_scatter_[idx].makeCompressed();
}

// ============================================================================
// CONDITIONS AUX LIMITES DIRICHLET
// ============================================================================

/**
 * @brief Applique les conditions de Dirichlet (vacuum) en modifiant la matrice A
 * 
 * Dans la formulation mixte-duale de Hébert:
 *   (1/D)∫ J·ψ dV + ∫ φ div(ψ) dV = ∫_∂Ω φ (ψ·n) dS
 * 
 * La condition de vacuum (Marshak, albedo β=0) impose:
 *   φ|_∂Ω = 2 · (J·n)|_∂Ω
 * 
 * (Note: φ = 2·(J·n), PAS 2D·(J·n). Car J = -D∇φ et Marshak donne
 *  φ + 2D·dφ/dn = 0 → φ - 2(J·n) = 0 → φ = 2(J·n))
 * 
 * Substituant dans le terme de bord:
 *   ∫_∂Ω φ (ψ·n) dS = 2 ∫_∂Ω (J·n)(ψ·n) dS
 * 
 * Ceci modifie A:
 *   A_eff(i,j) = A(i,j) - 2 · ∫_face (ψᵢ·n)(ψⱼ·n) dS
 * 
 * G est diagonal (orthogonalité Legendre des fonctions transverses).
 */
void NeutFEM::ApplyDirichletToA(int group) {
    const int n_J = fespace_.n_J;
    const int n_elem = mesh_.GetNE();
    const int nf = fespace_.dofs_per_face;
    
    std::vector<Triplet> triplets;
    
    // === Direction X ===
    {
        int attr_left = GetBoundaryAttribute(mesh_.dim, 0, false);
        int attr_right = GetBoundaryAttribute(mesh_.dim, 0, true);
        bool dirichlet_left = (bc_types_.count(attr_left) && bc_types_[attr_left] == BCType::DIRICHLET);
        bool dirichlet_right = (bc_types_.count(attr_right) && bc_types_[attr_right] == BCType::DIRICHLET);
        
        for (int iz = 0; iz < mesh_.nz; ++iz) {
            for (int iy = 0; iy < mesh_.ny; ++iy) {
                if (dirichlet_left) {
                    int elem = mesh_.ElemIndex(0, iy, iz);
                    double D = D_data_(group * n_elem + elem);
                    double fa = mesh_.FaceArea(elem, 0);
                    for (int f = 0; f < nf; ++f) {
                        int dof = fespace_.JxFaceIndex(0, iy, iz, f);
                        double G_ff = ComputeBoundaryFaceIntegral(f, fa) * 2.0 * D;
                        triplets.emplace_back(dof, dof, +G_ff);
                    }
                }
                if (dirichlet_right) {
                    int elem = mesh_.ElemIndex(mesh_.nx - 1, iy, iz);
                    double D = D_data_(group * n_elem + elem);
                    double fa = mesh_.FaceArea(elem, 0);
                    for (int f = 0; f < nf; ++f) {
                        int dof = fespace_.JxFaceIndex(mesh_.nx, iy, iz, f);
                        double G_ff = ComputeBoundaryFaceIntegral(f, fa) * 2.0 * D;
                        triplets.emplace_back(dof, dof, +G_ff);
                    }
                }
            }
        }
    }
    
    // === Direction Y (2D et 3D) ===
    if (mesh_.dim >= 2) {
        int attr_bottom = GetBoundaryAttribute(mesh_.dim, 1, false);
        int attr_top = GetBoundaryAttribute(mesh_.dim, 1, true);
        bool dirichlet_bottom = (bc_types_.count(attr_bottom) && bc_types_[attr_bottom] == BCType::DIRICHLET);
        bool dirichlet_top = (bc_types_.count(attr_top) && bc_types_[attr_top] == BCType::DIRICHLET);
        
        for (int iz = 0; iz < mesh_.nz; ++iz) {
            for (int ix = 0; ix < mesh_.nx; ++ix) {
                if (dirichlet_bottom) {
                    int elem = mesh_.ElemIndex(ix, 0, iz);
                    double D = D_data_(group * n_elem + elem);
                    double fa = mesh_.FaceArea(elem, 1);
                    for (int f = 0; f < nf; ++f) {
                        int dof = fespace_.JyFaceIndex(ix, 0, iz, f);
                        double G_ff = ComputeBoundaryFaceIntegral(f, fa) * 2.0* D;
                        triplets.emplace_back(dof, dof, +G_ff);
                    }
                }
                if (dirichlet_top) {
                    int elem = mesh_.ElemIndex(ix, mesh_.ny - 1, iz);
                    double D = D_data_(group * n_elem + elem);
                    double fa = mesh_.FaceArea(elem, 1);
                    for (int f = 0; f < nf; ++f) {
                        int dof = fespace_.JyFaceIndex(ix, mesh_.ny, iz, f);
                        double G_ff = ComputeBoundaryFaceIntegral(f, fa) * 2.0 * D;
                        triplets.emplace_back(dof, dof, +G_ff);
                    }
                }
            }
        }
    }
    
    // === Direction Z (3D) ===
    if (mesh_.dim == 3) {
        int attr_back = GetBoundaryAttribute(mesh_.dim, 2, false);
        int attr_front = GetBoundaryAttribute(mesh_.dim, 2, true);
        bool dirichlet_back = (bc_types_.count(attr_back) && bc_types_[attr_back] == BCType::DIRICHLET);
        bool dirichlet_front = (bc_types_.count(attr_front) && bc_types_[attr_front] == BCType::DIRICHLET);
        
        for (int iy = 0; iy < mesh_.ny; ++iy) {
            for (int ix = 0; ix < mesh_.nx; ++ix) {
                if (dirichlet_back) {
                    int elem = mesh_.ElemIndex(ix, iy, 0);
                    double D = D_data_(group * n_elem + elem);
                    double fa = mesh_.FaceArea(elem, 2);
                    for (int f = 0; f < nf; ++f) {
                        int dof = fespace_.JzFaceIndex(ix, iy, 0, f);
                        double G_ff = ComputeBoundaryFaceIntegral(f, fa) * 2.0 * D;
                        triplets.emplace_back(dof, dof, +G_ff);
                    }
                }
                if (dirichlet_front) {
                    int elem = mesh_.ElemIndex(ix, iy, mesh_.nz - 1);
                    double D = D_data_(group * n_elem + elem);
                    double fa = mesh_.FaceArea(elem, 2);
                    for (int f = 0; f < nf; ++f) {
                        int dof = fespace_.JzFaceIndex(ix, iy, mesh_.nz, f);
                        double G_ff = ComputeBoundaryFaceIntegral(f, fa) * 2.0 * D;
                        triplets.emplace_back(dof, dof, +G_ff);
                    }
                }
            }
        }
    }
    
    if (!triplets.empty()) {
        int modified = 0;
        for (const auto& t : triplets) {
            int row = t.row();
            int col = t.col();
            double val = t.value();
            // Modifier le coefficient directement dans la matrice creuse
            A_mats_[group].coeffRef(row, col) += val;
            modified++;
        }
        
        // Vérifier que la modification a pris effet
        if (!triplets.empty()) {
            int test_dof = triplets[0].row();
            double test_val = triplets[0].value();
        }
        
        A_mats_[group].makeCompressed();
        Log(VerbosityLevel::NORMAL, "    ", modified, " coefficients modifies");
    } else {
        Log(VerbosityLevel::NORMAL, "    AUCUN triplet genere! BC non appliquees.");
    }
}

/**
 * @brief Calcule ∫_face (ψ_f·n)² dS pour un DOF de face au bord
 * 
 * Piola contravariant: ψ_phys = (jac_d/det_J) * ψ_ref
 * Sur la face, ψ_ref = P_a(η) * P_b(ζ), donc :
 *   (ψ·n)_phys = (jac_d/det_J) * P_a(η) * P_b(ζ)
 * 
 * Pour direction x en 2D : (ψ·n)_phys = (2/hy) * P_a(η_ref)
 *   ∫_face (ψ·n)² dS = (4/hy²) * (hy/2) * ∫ P_a² dη = 2·mass_a/hy
 */
double NeutFEM::ComputeBoundaryFaceIntegral(int local_face_dof, double face_area) const {
    const int k = rt_order_int_;
    
    if (mesh_.dim == 1) {
        // 1D: face is a point, ψ·n = ±1, integral = 1
        return 1.0;
    }
    
    if (mesh_.dim == 2) {
        // 2D: 1 transverse index a
        int a = local_face_dof;
        double mass_a = 2.0 / (2.0 * a + 1.0);  // ∫_{-1}^{1} P_a² dξ
        return 2.0 * mass_a / face_area;
    }
    
    // 3D: 2 transverse indices (a, b)
    int a = local_face_dof % (k + 1);
    int b = local_face_dof / (k + 1);
    double mass_a = 2.0 / (2.0 * a + 1.0);
    double mass_b = 2.0 / (2.0 * b + 1.0);
    return 4.0 * mass_a * mass_b / face_area;
}

// ============================================================================
// MÉTHODES HELPER: FISSION SOURCE ET MATRICES PONDÉRÉES
// ============================================================================

SpMat NeutFEM::AssembleWeightedMassMatrix(const Vec_t& coeff_per_elem) const {
    const int n_Phi = fespace_.n_Phi;
    std::vector<Triplet> triplets;
    int n_Phi_local = local_matrices_->NumPhiDofs();
    triplets.reserve(mesh_.GetNE() * n_Phi_local * n_Phi_local);
    
    std::vector<int> Phi_indices;
    LocalMatrices* lm = const_cast<LocalMatrices*>(local_matrices_.get());
    
    for (int iz = 0; iz < mesh_.nz; ++iz) {
        for (int iy = 0; iy < mesh_.ny; ++iy) {
            for (int ix = 0; ix < mesh_.nx; ++ix) {
                const int e = mesh_.ElemIndex(ix, iy, iz);
                const double coeff = coeff_per_elem(e);
                if (std::abs(coeff) < 1e-14) continue;
                
                lm->Compute(e, 1.0, coeff);
                lm->GetGlobalPhiIndices(ix, iy, iz, Phi_indices);
                const Mat& C_loc = lm->GetC();
                
                for (int i = 0; i < n_Phi_local; ++i) {
                    for (int j = 0; j < n_Phi_local; ++j) {
                        if (std::abs(C_loc(i, j)) > 1e-14)
                            triplets.emplace_back(Phi_indices[i], Phi_indices[j], C_loc(i, j));
                    }
                }
            }
        }
    }
    
    SpMat M(n_Phi, n_Phi);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();
    return M;
}

/**
 * @brief Construit le RHS de fission pour le problème direct
 * 
 * Version vectorisée pour P0 (cas le plus fréquent)
 * 
 * Pour P0: rhs_g(e) += (χ_g(e) / keff) * total_fiss(e)
 * Pour P>=1: application bloc par bloc car χ est constant par maille
 */
void NeutFEM::BuildFissionRHS(int g, const Vec& total_fiss, double keff, Vec& group_rhs) const {
    const int n_elem = mesh_.GetNE();
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    const double inv_keff = 1.0 / keff;
    
    if (dofs_per_elem == 1) {
        // Cas P0 - utiliser opérations vectorielles Eigen
        // group_rhs += (Chi_g / keff) .* total_fiss
        Eigen::Map<const Vec> chi_g(Chi_data_.data() + g * n_elem, n_elem);
        group_rhs.noalias() += inv_keff * (chi_g.array() * total_fiss.array()).matrix();
    } else {
        // Cas P>=1: boucle sur les éléments (χ constant par maille)
        for (int e = 0; e < n_elem; ++e) {
            const double chi_val = Chi_data_(g * n_elem + e) * inv_keff;
            if (std::abs(chi_val) < 1e-14) continue;  // Skip éléments non-fissiles
            
            const int base_idx = e * dofs_per_elem;
            for (int d = 0; d < dofs_per_elem; ++d) {
                group_rhs(base_idx + d) += chi_val * total_fiss(base_idx + d);
            }
        }
    }
}

/**
 * @brief Construit le RHS de fission pour le problème adjoint
 * 
 * Version vectorisée similaire à BuildFissionRHS
 */
void NeutFEM::BuildFissionRHSAdjoint(int g, const Vec& total_chi_adj, double keff, Vec& group_rhs) const {
    const int n_elem = mesh_.GetNE();
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    const double inv_keff = 1.0 / keff;
    
    if (dofs_per_elem == 1) {
        // Cas P0 - opérations vectorielles
        Eigen::Map<const Vec> nsf_g(NSF_data_.data() + g * n_elem, n_elem);
        group_rhs.noalias() += inv_keff * (nsf_g.array() * total_chi_adj.array()).matrix();
    } else {
        // Cas P>=1
        for (int e = 0; e < n_elem; ++e) {
            const double nsf_val = NSF_data_(g * n_elem + e) * inv_keff;
            if (std::abs(nsf_val) < 1e-14) continue;
            
            const int base_idx = e * dofs_per_elem;
            for (int d = 0; d < dofs_per_elem; ++d) {
                group_rhs(base_idx + d) += nsf_val * total_chi_adj(base_idx + d);
            }
        }
    }
}

// ============================================================================
// RÉSOLUTION DU PROBLÈME DIRECT - CALCUL DE K-EFFECTIF
// ============================================================================
// 
// ALGORITHME DES ITÉRATIONS DE PUISSANCE:
// --------------------------------------
// Le problème aux valeurs propres de criticité s'écrit:
//   H φ = (1/k) F φ
// où H est l'opérateur de transport/diffusion et F l'opérateur de fission.
// 
// Itérations de puissance:
// 1. Initialiser φ⁰ = 1, k⁰ = 1
// 2. Calculer la source de fission: S_f = F φⁿ
// 3. Pour chaque groupe g, résoudre: H_g φ_g^{n+1} = (χ_g/kⁿ) S_f + scatter
// 4. Calculer la nouvelle production: P^{n+1} = <νΣf, φ^{n+1}>
// 5. Mettre à jour: k^{n+1} = kⁿ × (P^{n+1} / Pⁿ)
// 6. Normaliser φ^{n+1}
// 7. Répéter jusqu'à convergence
// 
// OPTIMISATIONS DISPONIBLES:
// -------------------------
// - use_diagonal_solver: Solveur diagonal pour RT0-P0 (très rapide, peu de RAM)
// - use_cmfd: Accélération CMFD (convergence rapide pour grands systèmes)
// - use_coarse_init: Initialisation multi-grille
// 
// ============================================================================

/**
 * @brief Calcule le facteur de multiplication effectif k-eff
 * 
 * @param use_coarse_init     Si true, initialise par résolution grossière
 * @param coarse_factors      Facteurs de réduction {rx, ry, rz}
 * @param use_diagonal_solver Si true, utilise le solveur diagonal RT0-P0 (économe en RAM)
 * @param use_cmfd            Si true, active l'accélération CMFD
 * @return Le facteur de multiplication effectif k-eff
 */
double NeutFEM::SolveKeff(bool use_coarse_init, const std::vector<int>& coarse_factors,
                          bool use_diagonal_solver, bool use_cmfd) {
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL DE K-EFFECTIF (DIRECT) ===");
    
    const int ng = num_groups_;
    const int n_Phi = fespace_.n_Phi;
    const int n_J = fespace_.n_J;
    
    // =========================================================================
    // CONFIGURATION DES OPTIMISATIONS
    // =========================================================================
    
    // Solveur diagonal: uniquement pour RT0-P0
    bool can_use_diagonal = (rt_order_int_ == 0 && p_order_int_ == 0);
    if (use_diagonal_solver && !can_use_diagonal) {
        Log(VerbosityLevel::NORMAL, "  Note: Solveur diagonal non disponible (ordre > 0)");
        use_diagonal_solver = false;
    }
    
    if (use_diagonal_solver) {
        Log(VerbosityLevel::NORMAL, "  Mode: Solveur diagonal RT0-P0 (faible RAM)");
        BuildDiagonalSchurCache();
    }
    
    if (use_cmfd) {
        Log(VerbosityLevel::NORMAL, "  Acceleration: CMFD active");
        InitializeCMFD();
    } else {
        Log(VerbosityLevel::NORMAL, "  Acceleration: Chebyshev");
    }
    
    // =========================================================================
    // INITIALISATION
    // =========================================================================
    
    double keff = has_valid_keff_ ? last_keff_direct_ : 1.0;
    
    // Initialisation multi-grille optionnelle
    if (use_coarse_init && !coarse_factors.empty()) {
        auto [keff_c, flux_c] = SolveCoarse(coarse_factors);
        Sol_Phi_ = flux_c;
        keff = keff_c;  // Utiliser le k-eff grossier comme point de départ
        Log(VerbosityLevel::NORMAL, "  k-eff initial (coarse) = ", std::fixed, std::setprecision(6), keff);
    }
    
    // Accélération de Chebyshev (utilisée si CMFD désactivé)
    ChebyshevAccel accel(15, 0.98);
    
    // Pré-allocation des vecteurs de travail
    Vec total_fiss(n_Phi);
    Vec group_rhs(n_Phi);
    Vec Sol_Phi_old(Sol_Phi_.size());
    
    // Vecteurs de contribution de fission par groupe
    std::vector<Vec> fiss_contrib(ng);
    for (int g = 0; g < ng; ++g) {
        fiss_contrib[g].resize(n_Phi);
    }
    
    double diff_k = 1.0;
    double diff_flux = 1.0;
    auto start = std::chrono::high_resolution_clock::now();
    
    // =========================================================================
    // BOUCLE PRINCIPALE D'ITÉRATIONS DE PUISSANCE
    // =========================================================================
    
    for (int it = 0; it < max_outer_iter_; ++it) {
        Sol_Phi_old = Sol_Phi_;
        
        // -----------------------------------------------------------------
        // CALCUL DE LA SOURCE DE FISSION TOTALE
        // -----------------------------------------------------------------
        total_fiss.setZero();
        for (int g = 0; g < ng; ++g) {
            Eigen::Map<const Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
            fiss_contrib[g].noalias() = M_fiss_[g] * Phi_g;
            total_fiss += fiss_contrib[g];
        }
        
        const double prod_old = total_fiss.sum();
        
        // -----------------------------------------------------------------
        // RÉSOLUTION GROUPE PAR GROUPE
        // -----------------------------------------------------------------
        for (int g = 0; g < ng; ++g) {
            group_rhs.setZero();
            
            // Source de fission: χ_g × total_fiss / keff
            BuildFissionRHS(g, total_fiss, keff, group_rhs);
            
            // Scattering inter-groupe
            for (int gp = 0; gp < ng; ++gp) {
                if (gp == g) continue;
                const int idx = g * ng + gp;
                if (M_scatter_[idx].nonZeros() == 0) continue;
                
                Eigen::Map<const Vec> Phi_gp(Sol_Phi_.data() + gp * n_Phi, n_Phi);
                group_rhs.noalias() += M_scatter_[idx] * Phi_gp;
            }
            
            // ---------------------------------------------------------
            // RÉSOLUTION DU SYSTÈME LINÉAIRE
            // ---------------------------------------------------------
            if (use_diagonal_solver) {
                // Solveur diagonal RT0-P0 (très rapide)
                Vec Phi_sol, J_sol;
                SolveDiagonalSchur(g, group_rhs, Phi_sol, J_sol);
                
                // Copier les solutions
                Eigen::Map<Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
                Eigen::Map<Vec> J_g(Sol_J_.data() + g * n_J, n_J);
                Phi_g = Phi_sol;
                J_g = J_sol;
            } else {
                // Solveur Schur standard (itératif)
                SolveGroupInternal(g, group_rhs);
            }
        }
        
        // -----------------------------------------------------------------
        // ACCÉLÉRATION CMFD (si activée)
        // -----------------------------------------------------------------
        if (use_cmfd && it >= 2) {
            // Mettre à jour les facteurs de correction D̂
            UpdateDhatCoefficients();
            
            // Appliquer la correction CMFD à chaque groupe
            for (int g = 0; g < ng; ++g) {
                Vec correction = ApplyCMFDCorrection(g, total_fiss, keff);
                
                Eigen::Map<Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
                Phi_g = Phi_g.array() * correction.array();
            }
        }
        
        // -----------------------------------------------------------------
        // MISE À JOUR DE K-EFF
        // -----------------------------------------------------------------
        double prod_new = 0.0;
        for (int g = 0; g < ng; ++g) {
            Eigen::Map<const Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
            prod_new += (M_fiss_[g] * Phi_g).sum();
        }
        
        const double keff_new = keff * (prod_new / prod_old);
        diff_k = std::abs(keff_new - keff);
        if (it >= 1) keff = keff_new;
        
        // Convergence du flux
        const double sol_norm_sq = Sol_Phi_.squaredNorm();
        const double diff_norm_sq = (Sol_Phi_ - Sol_Phi_old).squaredNorm();
        diff_flux = std::sqrt(diff_norm_sq / sol_norm_sq);
        
        // Normalisation
        const double norm = std::sqrt(sol_norm_sq);
        if (norm > 1e-14) Sol_Phi_ /= norm;

        // Accélération de Chebyshev (si CMFD désactivé)
        if (!use_cmfd && it >= 2) {
            accel(Sol_Phi_);
        }
       
        // Affichage
        if (verbosity_ >= VerbosityLevel::NORMAL && (it % 5 == 0)) {
            std::cout << "  It " << std::setw(4) << it
                      << " : k = " << std::fixed << std::setprecision(8) << keff
                      << "  dk = " << std::scientific << std::setprecision(2) << diff_k
                      << "  dphi = " << diff_flux << std::defaultfloat << std::endl;
        }
        
        // Test de convergence
        if (diff_k < tol_keff_ && diff_flux < tol_flux_) {
            Log(VerbosityLevel::NORMAL, "  Convergence en ", it + 1, " iterations");
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    has_valid_keff_ = true;
    last_keff_direct_ = keff;
    
    Log(VerbosityLevel::NORMAL, "  k-eff direct = ", std::fixed, std::setprecision(8), keff);
    Log(VerbosityLevel::NORMAL, "  Temps        = ", std::fixed, std::setprecision(2), elapsed, " s\n");
    
    return keff;
}

/**
 * @brief Version simplifiée de SolveKeff (compatibilité ascendante)
 */
double NeutFEM::SolveKeff(bool use_coarse_init, const std::vector<int>& coarse_factors) {
    // Appeler la version complète avec les optimisations par défaut
    // use_diagonal_solver = true pour RT0-P0 (détection automatique)
    // use_cmfd = false par défaut (Chebyshev plus simple)
    bool use_diagonal = (rt_order_int_ == 0 && p_order_int_ == 0);
    return SolveKeff(use_coarse_init, coarse_factors, use_diagonal, false);
}
    
// ============================================================================
// RÉSOLUTION DU PROBLÈME ADJOINT
// ============================================================================
// 
// THÉORIE DU PROBLÈME ADJOINT:
// ---------------------------
// L'équation adjointe de la diffusion neutronique est:
//   H† φ† = (1/k†) F† φ†
// où H† et F† sont les opérateurs adjoints (transposés) de H et F.
// 
// Pour la formulation mixte:
// - Les matrices A et C sont symétriques → A† = A, C† = C
// - La matrice B change de signe → B† = -Bᵀ, mais le système reste similaire
// - Les opérateurs de fission et scattering sont transposés:
//   * Direct:  source_g = χ_g × Σ_{g'} νΣf_{g'} φ_{g'}
//   * Adjoint: source_g = νΣf_g × Σ_{g'} χ_{g'} φ†_{g'}
//   * Scattering adjoint: Σ_{s,g'→g} devient Σ_{s,g→g'}
// 
// PROPRIÉTÉS:
// ----------
// - Le k-eff adjoint doit être égal au k-eff direct (k† = k)
// - Les flux direct et adjoint sont bi-orthogonaux: <φ, F† φ†> peut servir
//   de normalisation
// - Utilisations: sensibilités, calculs de perturbation, importance
// 
// ============================================================================

/**
 * @brief Résout le problème aux valeurs propres adjoint
 * 
 * Cette méthode calcule le flux adjoint φ† et le k-eff adjoint k† par
 * itérations de puissance inverses sur le système adjoint transposé.
 * 
 * ALGORITHME:
 * ----------
 * 1. Si use_direct_keff=true et qu'un k-eff direct existe, on fixe k† = k
 *    et on résout uniquement pour φ† (mode source fixe)
 * 2. Sinon, on fait des itérations de puissance complètes pour trouver k†
 * 
 * La source de fission adjointe pour le groupe g est:
 *   S†_g = (νΣf_g / k†) × Σ_{g'} χ_{g'} φ†_{g'}
 * 
 * Le scattering adjoint est transposé:
 *   Σ†_{s,g'→g} φ†_{g'} = Σ_{s,g→g'} φ†_{g'}
 * 
 * @param normalize_to_direct  Si true, normalise φ† tel que <φ, φ†> = 1
 * @param use_direct_keff      Si true, utilise le k-eff direct au lieu de calculer k†
 * @return Le k-eff adjoint calculé
 */
double NeutFEM::SolveAdjoint(bool normalize_to_direct, bool use_direct_keff) {
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL DE K-EFFECTIF (ADJOINT) ===");
    
    const int ng = num_groups_;
    const int n_Phi = fespace_.n_Phi;
    const int n_elem = mesh_.GetNE();
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    
    // Utiliser k-eff direct si disponible et demandé
    double keff_adj = 1.0;
    if (use_direct_keff && has_valid_keff_) {
        keff_adj = last_keff_direct_;
        Log(VerbosityLevel::NORMAL, "  Utilisation k-eff direct = ", 
            std::fixed, std::setprecision(8), keff_adj);
    }
    
    // Initialisation du flux adjoint
    Sol_Phi_adj_.setConstant(1.0);
    Sol_Phi_adj_ /= Sol_Phi_adj_.norm();
    
    ChebyshevAccel accel(15, 0.98);
    Vec group_rhs(n_Phi);
    Vec Sol_Phi_adj_old = Sol_Phi_adj_;
    
    // Pré-calcul de la somme des νΣf par élément
    Vec_t total_nsf_per_elem(n_elem);
    for (int e = 0; e < n_elem; ++e) {
        double sum = 0.0;
        for (int g = 0; g < ng; ++g) {
            sum += NSF_data_(g * n_elem + e);
        }
        total_nsf_per_elem(e) = sum;
    }
    
    double diff_k = 1.0;
    double diff_flux = 1.0;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int it = 0; it < max_outer_iter_; ++it) {
        Sol_Phi_adj_old = Sol_Phi_adj_;
        
        // 1. Calcul de total_chi_adj = Σ_g M_chi[g] × φ†_g
        Vec total_chi_adj(n_Phi);
        total_chi_adj.setZero();
        for (int g = 0; g < ng; ++g) {
            Eigen::Map<const Vec> Phi_adj_g(Sol_Phi_adj_.data() + g * n_Phi, n_Phi);
            total_chi_adj += M_chi_[g] * Phi_adj_g;
        }
        
        // 2. Calcul de la production (pour mise à jour de k)
        // P = Σ_e (Σ_g νΣf_g,e) × [total_chi_adj]_e
        // Note: PAS de multiplication par le volume car M_chi inclut déjà le volume
        double prod_old = 0.0;
        for (int e = 0; e < n_elem; ++e) {
            prod_old += total_nsf_per_elem(e) * total_chi_adj(e * dofs_per_elem);
        }
        
        // 3. Résolution groupe par groupe
        for (int g = 0; g < ng; ++g) {
            group_rhs.setZero();
            
            // Source de fission adjointe: (νΣf_g / k†) × total_chi_adj
            BuildFissionRHSAdjoint(g, total_chi_adj, keff_adj, group_rhs);
            
            // Scattering adjoint: matrice transposée
            // Direct: M_scatter_[g*ng+gp] = Σ_{s,gp→g}
            // Adjoint: on veut Σ_{s,g→gp}, donc idx = gp*ng+g
            for (int gp = 0; gp < ng; ++gp) {
                if (gp == g) continue;
                const int idx = gp * ng + g;
                if (M_scatter_[idx].nonZeros() == 0) continue;
                Eigen::Map<const Vec> Phi_adj_gp(Sol_Phi_adj_.data() + gp * n_Phi, n_Phi);
                group_rhs += M_scatter_[idx] * Phi_adj_gp;
            }
            
            SolveGroupInternalAdjoint(g, group_rhs);
        }
        
        // 4. Calcul de la nouvelle production
        Vec total_chi_adj_new(n_Phi);
        total_chi_adj_new.setZero();
        for (int g = 0; g < ng; ++g) {
            Eigen::Map<const Vec> Phi_adj_g(Sol_Phi_adj_.data() + g * n_Phi, n_Phi);
            total_chi_adj_new += M_chi_[g] * Phi_adj_g;
        }

        double prod_new = 0.0;
        for (int e = 0; e < n_elem; ++e) {
            prod_new += total_nsf_per_elem(e) * total_chi_adj_new(e * dofs_per_elem);
        }
        
        // 5. Mise à jour k-eff adjoint
        double keff_new = keff_adj;
        
        if (!use_direct_keff || !has_valid_keff_) {
            // Mode itérations de puissance
            if (std::abs(prod_old) > 1e-14 && it > 0) {
                keff_new = keff_adj * (prod_new / prod_old);
            }
            diff_k = std::abs(keff_new - keff_adj);
            keff_adj = keff_new;
        } else {
            diff_k = 0.0;
        }
        
        // Convergence du flux
        diff_flux = (Sol_Phi_adj_ - Sol_Phi_adj_old).norm() / Sol_Phi_adj_.norm();
        
        // Normalisation
        const double norm = Sol_Phi_adj_.norm();
        if (norm > 1e-14) Sol_Phi_adj_ /= norm;
        
        // Accélération de Chebyshev (seulement en mode power iteration)
        if (!use_direct_keff && it >= 5) {
            accel(Sol_Phi_adj_);
        }

        // Affichage
        if (verbosity_ >= VerbosityLevel::NORMAL && (it % 5 == 0)) {
            std::cout << "  It " << std::setw(4) << it
                      << " : k† = " << std::fixed << std::setprecision(8) << keff_adj
                      << "  dk = " << std::scientific << std::setprecision(2) << diff_k
                      << "  dphi = " << diff_flux << std::defaultfloat << std::endl;
        }
        
        // Test de convergence
        bool converged = (diff_flux < tol_flux_);
        if (!use_direct_keff) {
            converged = converged && (diff_k < tol_keff_);
        }
        
        if (converged) {
            Log(VerbosityLevel::NORMAL, "  Convergence en ", it + 1, " iterations");
            break;
        }
    }

    // =========================================================================
    // NORMALISATION BI-ORTHOGONALE
    // =========================================================================
    // On normalise φ† tel que <φ, φ†>_M = 1 où M est la matrice de masse
    // avec les poids de Legendre appropriés pour P >= 1.
    
    if (normalize_to_direct && has_valid_keff_) {
        double inner_product = 0.0;
        
        for (int g = 0; g < ng; ++g) {
            Eigen::Map<const Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
            Eigen::Map<const Vec> Phi_adj_g(Sol_Phi_adj_.data() + g * n_Phi, n_Phi);
            
            for (int e = 0; e < n_elem; ++e) {
                double vol = mesh_.ElemVolume(e);
                
                for (int d = 0; d < dofs_per_elem; ++d) {
                    int local_idx = e * dofs_per_elem + d;
                    
                    // Décomposition de l'index DOF en indices Legendre (i, j, k)
                    int n_pk = p_order_int_ + 1;
                    int i_idx, j_idx, k_idx;
                    
                    if (mesh_.dim == 1) { 
                        i_idx = d; 
                        j_idx = 0; 
                        k_idx = 0; 
                    } else if (mesh_.dim == 2) { 
                        i_idx = d % n_pk; 
                        j_idx = d / n_pk; 
                        k_idx = 0; 
                    } else { 
                        i_idx = d % n_pk; 
                        j_idx = (d / n_pk) % n_pk; 
                        k_idx = d / (n_pk * n_pk); 
                    }
                    
                    // Poids de quadrature: ∫_{-1}^{1} P_n² dξ = 2/(2n+1)
                    // Normalisé pour l'intervalle physique
                    double w = Legendre::MassIntegral(i_idx, i_idx) / 2.0;
                    if (mesh_.dim >= 2) w *= Legendre::MassIntegral(j_idx, j_idx) / 2.0;
                    if (mesh_.dim >= 3) w *= Legendre::MassIntegral(k_idx, k_idx) / 2.0;
                    
                    inner_product += Phi_g(local_idx) * Phi_adj_g(local_idx) * vol * w;
                }
            }
        }
        
        if (std::abs(inner_product) > 1e-14) {
            Sol_Phi_adj_ /= inner_product;
            Log(VerbosityLevel::NORMAL, "  Normalisation bi-orthogonale <phi, phi†> = 1");
        }
    }
    
    // =========================================================================
    // FINALISATION
    // =========================================================================
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    has_valid_adjoint_ = true;
    last_keff_adjoint_ = keff_adj;
    
    Log(VerbosityLevel::NORMAL, "  k-eff adjoint = ", std::fixed, std::setprecision(8), keff_adj);
    Log(VerbosityLevel::NORMAL, "  Temps         = ", std::fixed, std::setprecision(2), elapsed, " s\n");
    
    return keff_adj;
}

void NeutFEM::SolveGroupInternal(int g, const Vec& source) {
    const int n_J = fespace_.n_J;
    const int n_Phi = fespace_.n_Phi;
    
    Eigen::Map<Vec> J_g(Sol_J_.data() + g * n_J, n_J);
    Eigen::Map<Vec> Phi_g(Sol_Phi_.data() + g * n_Phi, n_Phi);
    
    Vec rhs = source;
    ApplyBoundaryConditions(g, rhs);
    
    // Réutiliser la factorisation si les matrices n'ont pas changé
    // Le SchurSolver devrait détecter si SetMatrices reçoit les mêmes matrices
    // et éviter de re-factoriser (à implémenter dans SchurSolver)
    schur_solver_->SetMatrices(A_mats_[g], B_mat_, C_mats_[g]);
    schur_solver_->SetVerbose(verbosity_ >= VerbosityLevel::DEBUG);
    
    Vec J_sol, Phi_sol;
    schur_solver_->Solve(rhs, J_sol, Phi_sol);
    
    J_g = J_sol;
    Phi_g = Phi_sol;
}

void NeutFEM::SolveGroupInternalAdjoint(int g, const Vec& source) {
    const int n_J = fespace_.n_J;
    const int n_Phi = fespace_.n_Phi;
    
    Eigen::Map<Vec> J_adj_g(Sol_J_adj_.data() + g * n_J, n_J);
    Eigen::Map<Vec> Phi_adj_g(Sol_Phi_adj_.data() + g * n_Phi, n_Phi);
    
    Vec rhs = source;
    ApplyBoundaryConditions(g, rhs);
    
    // Les matrices A et C sont symétriques, identiques pour l'adjoint
    schur_solver_->SetMatrices(A_mats_[g], B_mat_, C_mats_[g]);
    schur_solver_->SetVerbose(verbosity_ >= VerbosityLevel::DEBUG);
    
    Vec J_sol, Phi_sol;
    schur_solver_->Solve(rhs, J_sol, Phi_sol);
    
    J_adj_g = J_sol;
    Phi_adj_g = Phi_sol;
}

void NeutFEM::ApplyBoundaryConditions(int /*group*/, Vec& /*rhs*/) {
    // Les conditions MIRROR sont naturelles pour RT (J·n = 0)
    // Les autres conditions nécessitent une modification du système
}

// ============================================================================
// EXPORT VTK
// ============================================================================

void NeutFEM::ExportVTK(const std::string& filename,
                        bool export_flux,
                        bool export_current,
                        bool export_xs,
                        bool export_adjoint) {
    
    std::string full_filename = filename + ".vtk";
    std::ofstream file(full_filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + full_filename);
    }
    
    Log(VerbosityLevel::NORMAL, "Export VTK vers ", full_filename);
    
    const int nx = mesh_.nx;
    const int ny = mesh_.ny;
    const int nz = mesh_.nz;
    const int n_cells = mesh_.GetNE();
    const int n_points = (nx + 1) * (ny + 1) * (nz + 1);
    
    // ---- Header VTK ----
    file << "# vtk DataFile Version 3.0\n";
    file << "NeutFEM Output - k-eff=" << std::fixed << std::setprecision(6) << last_keff_direct_ << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " " << (nz + 1) << "\n";
    
    // ---- Points ----
    file << "POINTS " << n_points << " double\n";
    for (int iz = 0; iz <= nz; ++iz) {
        double z = (mesh_.dim == 3) ? mesh_.z_breaks(iz) : 0.0;
        for (int iy = 0; iy <= ny; ++iy) {
            double y = (mesh_.dim >= 2) ? mesh_.y_breaks(iy) : 0.0;
            for (int ix = 0; ix <= nx; ++ix) {
                double x = mesh_.x_breaks(ix);
                file << x << " " << y << " " << z << "\n";
            }
        }
    }
    
    // ---- Cell data ----
    file << "\nCELL_DATA " << n_cells << "\n";
    
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    
    // ---- Flux par groupe (P0 component = cell average) ----
    if (export_flux) {
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS Flux_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << Sol_Phi_(g * fespace_.n_Phi + e * dofs_per_elem) << "\n";
            }
        }
        
        // Flux total
        file << "SCALARS Flux_total double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int e = 0; e < n_cells; ++e) {
            double total = 0.0;
            for (int g = 0; g < num_groups_; ++g) {
                total += Sol_Phi_(g * fespace_.n_Phi + e * dofs_per_elem);
            }
            file << total << "\n";
        }
    }
    
    // ---- Flux adjoint ----
    if (export_adjoint && has_valid_adjoint_) {
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS Flux_adj_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << Sol_Phi_adj_(g * fespace_.n_Phi + e * dofs_per_elem) << "\n";
            }
        }
    }
    
    // ---- Courant (vecteur) ----
    if (export_current) {
        for (int g = 0; g < num_groups_; ++g) {
            file << "VECTORS Current_g" << g << " double\n";
            for (int iz = 0; iz < nz; ++iz) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        // Courant moyen dans la cellule
                        int fl_x = fespace_.JxFaceIndex(ix, iy, iz);
                        int fr_x = fespace_.JxFaceIndex(ix + 1, iy, iz);
                        double Jx = 0.5 * (Sol_J_(g * fespace_.n_J + fl_x) + 
                                           Sol_J_(g * fespace_.n_J + fr_x));
                        
                        double Jy = 0.0;
                        double Jz = 0.0;
                        
                        if (mesh_.dim >= 2) {
                            int fb_y = fespace_.JyFaceIndex(ix, iy, iz);
                            int ft_y = fespace_.JyFaceIndex(ix, iy + 1, iz);
                            Jy = 0.5 * (Sol_J_(g * fespace_.n_J + fb_y) + 
                                        Sol_J_(g * fespace_.n_J + ft_y));
                        }
                        
                        if (mesh_.dim == 3) {
                            int fk_z = fespace_.JzFaceIndex(ix, iy, iz);
                            int ff_z = fespace_.JzFaceIndex(ix, iy, iz + 1);
                            Jz = 0.5 * (Sol_J_(g * fespace_.n_J + fk_z) + 
                                        Sol_J_(g * fespace_.n_J + ff_z));
                        }
                        
                        file << Jx << " " << Jy << " " << Jz << "\n";
                    }
                }
            }
        }
    }
    
    // ---- Sections efficaces ----
    if (export_xs) {
        // Coefficient de diffusion
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS D_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << D_data_(g * n_cells + e) << "\n";
            }
        }
        
        // Section de réaction
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS SigmaR_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << SigR_data_(g * n_cells + e) << "\n";
            }
        }
        
        // Nu-Sigma-fission
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS NuSigF_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << NSF_data_(g * n_cells + e) << "\n";
            }
        }
        
        // Spectre de fission
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS Chi_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << Chi_data_(g * n_cells + e) << "\n";
            }
        }
        
        // Kappa-Sigma-fission (puissance)
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS KappaSigF_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << KSF_data_(g * n_cells + e) << "\n";
            }
        }
        
        // Source externe
        for (int g = 0; g < num_groups_; ++g) {
            file << "SCALARS Source_g" << g << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int e = 0; e < n_cells; ++e) {
                file << SRC_data_(g * n_cells + e) << "\n";
            }
        }
        
        // Matrices de scattering
        for (int gf = 0; gf < num_groups_; ++gf) {
            for (int gt = 0; gt < num_groups_; ++gt) {
                int offset = GetSigSOffset(gf, gt);
                file << "SCALARS SigS_" << gf << "_to_" << gt << " double 1\n";
                file << "LOOKUP_TABLE default\n";
                for (int e = 0; e < n_cells; ++e) {
                    file << SigS_data_(offset + e) << "\n";
                }
            }
        }
    }
    
    file.close();
    Log(VerbosityLevel::NORMAL, "  Export termine: ", n_cells, " cellules");
}

void NeutFEM::ExportFluxVTK(const std::string& filename, bool adjoint) {
    ExportVTK(filename, true, false, false, adjoint);
}

void NeutFEM::ExportXSVTK(const std::string& filename) {
    ExportVTK(filename, false, false, true, false);
}

// ============================================================================
// MÉTHODES AUXILIAIRES
// ============================================================================

int NeutFEM::GetBoundaryAttribute(int dim, int direction, bool is_upper) const {
    if (dim == 1) return is_upper ? 2 : 1;
    if (dim == 2) {
        if (direction == 0) return is_upper ? 2 : 1;
        return is_upper ? 3 : 4;
    }
    if (direction == 0) return is_upper ? 4 : 3;
    if (direction == 1) return is_upper ? 5 : 6;
    return is_upper ? 2 : 1;
}

/**
 * @brief Résolution sur maillage grossier RT0-P0 pour initialisation multi-grille
 * 
 * Cette méthode implémente une stratégie de pré-calcul sur un maillage réduit
 * pour fournir une initialisation de qualité aux itérations de puissance.
 * 
 * ALGORITHME:
 * ----------
 * 1. Construire un maillage grossier en regroupant les mailles fines
 *    - refine = {rx, ry, rz}: facteurs de réduction par direction
 *    - nx_coarse = nx_fine / rx, etc.
 * 
 * 2. Homogénéiser les sections efficaces par moyenne volumique:
 *    - D_coarse = Σᵢ(Vᵢ × Dᵢ) / Σᵢ(Vᵢ)
 *    - Σ_coarse = Σᵢ(Vᵢ × Σᵢ) / Σᵢ(Vᵢ)
 * 
 * 3. Résoudre le problème aux valeurs propres sur le maillage grossier
 *    en RT0-P0 (rapide et robuste)
 * 
 * 4. Projeter la solution grossière sur le maillage fin comme initialisation
 * 
 * AVANTAGES:
 * ---------
 * - Réduction significative du temps de calcul des premières itérations
 * - Amélioration de la convergence pour les grands maillages
 * - Estimation rapide de k-eff avant le calcul fin
 * 
 * @param refine  Vecteur des facteurs de réduction {rx, ry, rz}
 *                Par exemple {2, 2, 1} regroupe 2×2×1 = 4 mailles en 2D
 * @return Paire (k-eff grossier, vecteur flux grossier projeté sur maillage fin)
 */
std::pair<double, Vec_t> NeutFEM::SolveCoarse(const std::vector<int>& refine) {
    
    // =========================================================================
    // 1. VALIDATION ET EXTRACTION DES PARAMÈTRES
    // =========================================================================
    
    if (refine.empty()) {
        Log(VerbosityLevel::NORMAL, "  SolveCoarse: facteurs de reduction vides, pas de calcul grossier");
        return {1.0, Sol_Phi_};
    }
    
    const int ng = num_groups_;
    const int dim = mesh_.dim;
    const int n_elem_fine = mesh_.GetNE();
    
    // Extraire les facteurs de réduction avec valeurs par défaut
    // rx = facteur de réduction en X (nombre de mailles fines par maille grossière)
    const int rx = (refine.size() > 0) ? std::max(refine[0], 1) : 1;
    const int ry = (refine.size() > 1 && dim >= 2) ? std::max(refine[1], 1) : 1;
    const int rz = (refine.size() > 2 && dim >= 3) ? std::max(refine[2], 1) : 1;
    
    // Vérifier que les facteurs divisent exactement le maillage fin
    if (mesh_.nx % rx != 0 || mesh_.ny % ry != 0 || mesh_.nz % rz != 0) {
        Log(VerbosityLevel::NORMAL, "  SolveCoarse: facteurs (", rx, ",", ry, ",", rz, 
            ") ne divisent pas le maillage (", mesh_.nx, ",", mesh_.ny, ",", mesh_.nz, ")");
        Log(VerbosityLevel::NORMAL, "  -> Pas de calcul grossier, utilisation du flux initial");
        return {1.0, Sol_Phi_};
    }
    
    // Dimensions du maillage grossier
    const int nx_c = mesh_.nx / rx;
    const int ny_c = mesh_.ny / ry;
    const int nz_c = mesh_.nz / rz;
    const int n_elem_coarse = nx_c * ny_c * nz_c;
    
    Log(VerbosityLevel::NORMAL, "\n=== RESOLUTION SUR MAILLAGE GROSSIER ===");
    Log(VerbosityLevel::NORMAL, "  Maillage fin    : ", mesh_.nx, " x ", mesh_.ny, " x ", mesh_.nz);
    Log(VerbosityLevel::NORMAL, "  Facteurs reduc. : ", rx, " x ", ry, " x ", rz);
    Log(VerbosityLevel::NORMAL, "  Maillage grossier: ", nx_c, " x ", ny_c, " x ", nz_c);
    
    // =========================================================================
    // 2. CONSTRUCTION DES BREAKS DU MAILLAGE GROSSIER
    // =========================================================================
    // Les breaks grossiers sont des sous-ensembles des breaks fins
    // x_breaks_coarse[i] = x_breaks_fine[i * rx]
    
    Vec_t x_breaks_c(nx_c + 1);
    Vec_t y_breaks_c(dim >= 2 ? ny_c + 1 : 1);
    Vec_t z_breaks_c(dim >= 3 ? nz_c + 1 : 1);
    
    for (int i = 0; i <= nx_c; ++i) {
        x_breaks_c(i) = mesh_.x_breaks(i * rx);
    }
    
    if (dim >= 2) {
        for (int j = 0; j <= ny_c; ++j) {
            y_breaks_c(j) = mesh_.y_breaks(j * ry);
        }
    } else {
        y_breaks_c(0) = 0.0;
    }
    
    if (dim >= 3) {
        for (int k = 0; k <= nz_c; ++k) {
            z_breaks_c(k) = mesh_.z_breaks(k * rz);
        }
    } else {
        z_breaks_c(0) = 0.0;
    }
    
    // =========================================================================
    // 3. CRÉATION DU SOLVEUR GROSSIER (RT0-P0)
    // =========================================================================
    // On utilise toujours RT0-P0 pour le maillage grossier car:
    // - C'est le plus rapide et le plus robuste
    // - L'ordre élevé n'apporte pas de précision sur maillage grossier
    // - Le but est juste d'avoir une bonne initialisation
    
    NeutFEM coarse_solver(0, ng, x_breaks_c, y_breaks_c, z_breaks_c);
    coarse_solver.SetLinearSolver(linear_solver_type_);
    coarse_solver.SetTolerance(tol_keff_ * 10.0, tol_flux_ * 10.0, tol_L2_, 
                               max_outer_iter_ / 2, max_inner_iter_);
    coarse_solver.SetVerbosity(VerbosityLevel::SILENT);
    
    // Copier les conditions aux limites
    for (const auto& [attr, type] : bc_types_) {
        coarse_solver.SetBC(attr, type, bc_values_.count(attr) ? bc_values_.at(attr) : 0.0);
    }
    
    // =========================================================================
    // 4. HOMOGÉNÉISATION DES SECTIONS EFFICACES
    // =========================================================================
    // Moyenne volumique pondérée sur chaque macro-maille:
    //   <Σ>_C = ∫_C Σ(r) dV / ∫_C dV = Σᵢ∈C (Σᵢ × Vᵢ) / Σᵢ∈C Vᵢ
    // 
    // Pour D (coefficient de diffusion), on utilise la moyenne harmonique
    // pour mieux représenter les courants à travers les interfaces:
    //   <D>_C = V_C / Σᵢ∈C (Vᵢ / Dᵢ)
    // En pratique, la moyenne arithmétique fonctionne aussi pour l'initialisation.
    
    // Lambda pour calculer l'indice d'élément fin à partir des indices (ix, iy, iz)
    auto fine_elem_index = [&](int ix, int iy, int iz) -> int {
        return iz * (mesh_.ny * mesh_.nx) + iy * mesh_.nx + ix;
    };
    
    // Lambda pour calculer le volume d'un élément fin
    auto fine_elem_volume = [&](int ix, int iy, int iz) -> double {
        double vol = mesh_.x_breaks(ix + 1) - mesh_.x_breaks(ix);
        if (dim >= 2) vol *= mesh_.y_breaks(iy + 1) - mesh_.y_breaks(iy);
        if (dim >= 3) vol *= mesh_.z_breaks(iz + 1) - mesh_.z_breaks(iz);
        return vol;
    };
    
    // Pour chaque maille grossière, agréger les mailles fines
    for (int g = 0; g < ng; ++g) {
        for (int kz_c = 0; kz_c < nz_c; ++kz_c) {
            for (int ky_c = 0; ky_c < ny_c; ++ky_c) {
                for (int kx_c = 0; kx_c < nx_c; ++kx_c) {
                    
                    // Indice de l'élément grossier
                    const int e_c = kz_c * (ny_c * nx_c) + ky_c * nx_c + kx_c;
                    
                    // Accumulateurs pour la moyenne volumique
                    double vol_total = 0.0;
                    double sum_D = 0.0;
                    double sum_SigR = 0.0;
                    double sum_NSF = 0.0;
                    double sum_KSF = 0.0;
                    double sum_Chi = 0.0;
                    std::vector<double> sum_SigS(ng, 0.0);
                    
                    // Parcourir les mailles fines de cette macro-maille
                    for (int sz = 0; sz < rz; ++sz) {
                        for (int sy = 0; sy < ry; ++sy) {
                            for (int sx = 0; sx < rx; ++sx) {
                                
                                // Indices fins
                                int ix_f = kx_c * rx + sx;
                                int iy_f = ky_c * ry + sy;
                                int iz_f = kz_c * rz + sz;
                                
                                int e_f = fine_elem_index(ix_f, iy_f, iz_f);
                                double vol_f = fine_elem_volume(ix_f, iy_f, iz_f);
                                
                                // Accumulation pondérée par le volume
                                vol_total += vol_f;
                                sum_D += vol_f * D_data_(g * n_elem_fine + e_f);
                                sum_SigR += vol_f * SigR_data_(g * n_elem_fine + e_f);
                                sum_NSF += vol_f * NSF_data_(g * n_elem_fine + e_f);
                                sum_KSF += vol_f * KSF_data_(g * n_elem_fine + e_f);
                                sum_Chi += vol_f * Chi_data_(g * n_elem_fine + e_f);
                                
                                // Scattering pour tous les groupes sources
                                for (int gp = 0; gp < ng; ++gp) {
                                    int off_f = GetSigSOffset(gp, g);
                                    sum_SigS[gp] += vol_f * SigS_data_(off_f + e_f);
                                }
                            }
                        }
                    }
                    
                    // Calculer les moyennes et les affecter au solveur grossier
                    // Note: vol_total > 0 garanti car rx, ry, rz >= 1
                    coarse_solver.D_data_(g * n_elem_coarse + e_c) = sum_D / vol_total;
                    coarse_solver.SigR_data_(g * n_elem_coarse + e_c) = sum_SigR / vol_total;
                    coarse_solver.NSF_data_(g * n_elem_coarse + e_c) = sum_NSF / vol_total;
                    coarse_solver.KSF_data_(g * n_elem_coarse + e_c) = sum_KSF / vol_total;
                    coarse_solver.Chi_data_(g * n_elem_coarse + e_c) = sum_Chi / vol_total;
                    
                    for (int gp = 0; gp < ng; ++gp) {
                        int off_c = coarse_solver.GetSigSOffset(gp, g);
                        coarse_solver.SigS_data_(off_c + e_c) = sum_SigS[gp] / vol_total;
                    }
                }
            }
        }
    }
    
    // =========================================================================
    // 5. ASSEMBLAGE ET RÉSOLUTION SUR MAILLAGE GROSSIER
    // =========================================================================
    
    Log(VerbosityLevel::NORMAL, "  Assemblage matrices grossieres...");
    coarse_solver.BuildMatrices();
    
    Log(VerbosityLevel::NORMAL, "  Resolution k-eff grossier...");
    // Utiliser la version complète avec solveur diagonal DÉSACTIVÉ pour éviter les problèmes
    // Le solveur grossier utilise le solveur itératif standard
    double keff_coarse = coarse_solver.SolveKeff(false, {}, false, false);
    
    Log(VerbosityLevel::NORMAL, "  k-eff grossier = ", std::fixed, std::setprecision(6), keff_coarse);
    
    // =========================================================================
    // 6. PROJECTION DU FLUX GROSSIER SUR LE MAILLAGE FIN
    // =========================================================================
    // Chaque maille fine hérite de la valeur de sa maille parente.
    // Pour P0, c'est une simple recopie. Pour P>0, on initialiserait
    // seulement le mode constant (DOF 0).
    
    const int n_Phi_fine = fespace_.n_Phi;
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    
    Vec_t flux_projected(ng * n_Phi_fine);
    flux_projected.setZero();
    
    for (int g = 0; g < ng; ++g) {
        for (int iz_f = 0; iz_f < mesh_.nz; ++iz_f) {
            for (int iy_f = 0; iy_f < mesh_.ny; ++iy_f) {
                for (int ix_f = 0; ix_f < mesh_.nx; ++ix_f) {
                    
                    // Indice fin et indice grossier correspondant
                    int e_f = iz_f * (mesh_.ny * mesh_.nx) + iy_f * mesh_.nx + ix_f;
                    int ix_c = ix_f / rx;
                    int iy_c = iy_f / ry;
                    int iz_c = iz_f / rz;
                    int e_c = iz_c * (ny_c * nx_c) + iy_c * nx_c + ix_c;
                    
                    // Valeur du flux grossier (P0 = 1 DOF par élément)
                    double phi_coarse = coarse_solver.Sol_Phi_(g * n_elem_coarse + e_c);
                    
                    // Affecter au DOF 0 (mode constant) de la maille fine
                    // Les autres DOFs (modes d'ordre supérieur) restent à zéro
                    flux_projected(g * n_Phi_fine + e_f * dofs_per_elem) = phi_coarse;
                }
            }
        }
    }
    
    Log(VerbosityLevel::NORMAL, "  Projection flux grossier -> fin terminee\n");
    
    return {keff_coarse, flux_projected};
}


int NeutFEM::AddReflector(py::array_t<double>, py::array_t<double>, py::array_t<double>) {
    return 0;
}

void NeutFEM::SetReflector(int, int, bool) {}
void NeutFEM::ClearReflectors() {}
void NeutFEM::SelectOptimalSolver() {}

// ============================================================================
// ACCESSEURS PYTHON
// ============================================================================

py::array_t<double> make_numpy_array(Vec_t& data, int ng, int nx, int ny, int nz,
                                     int dim, py::object owner) {
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    
    shape.push_back(ng);
    if (dim >= 3) shape.push_back(nz);
    if (dim >= 2) shape.push_back(ny);
    shape.push_back(nx);
    
    py::ssize_t stride = sizeof(double);
    strides.resize(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    
    return py::array_t<double>(shape, strides, data.data(), owner);
}

py::array_t<double> NeutFEM::py_get_D() {
    return make_numpy_array(D_data_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_SRC() {
    return make_numpy_array(SRC_data_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_SigR() {
    return make_numpy_array(SigR_data_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_NSF() {
    return make_numpy_array(NSF_data_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_KSF() {
    return make_numpy_array(KSF_data_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_Chi() {
    return make_numpy_array(Chi_data_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_SigS() {
    const int ng = num_groups_;
    const int ne = mesh_.GetNE();
    
    std::vector<py::ssize_t> shape = {ng, ng};
    if (mesh_.dim >= 3) shape.push_back(mesh_.nz);
    if (mesh_.dim >= 2) shape.push_back(mesh_.ny);
    shape.push_back(mesh_.nx);
    
    py::ssize_t stride = sizeof(double);
    std::vector<py::ssize_t> strides(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    
    return py::array_t<double>(shape, strides, SigS_data_.data(), py::cast(this));
}


py::array_t<double> NeutFEM::py_get_flux() {
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    if (dofs_per_elem == 1) {
        // P0: direct mapping
        return make_numpy_array(Sol_Phi_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                                mesh_.dim, py::cast(this));
    }
    // P>=1: extract P0 component (cell average) for each element
    const int n_elem = mesh_.GetNE();
    flux_P0_.resize(num_groups_ * n_elem);
    for (int g = 0; g < num_groups_; ++g) {
        for (int e = 0; e < n_elem; ++e) {
            flux_P0_(g * n_elem + e) = Sol_Phi_(g * fespace_.n_Phi + e * dofs_per_elem);
        }
    }
    return make_numpy_array(flux_P0_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}

py::array_t<double> NeutFEM::py_get_flux_adj() {
    const int dofs_per_elem = fespace_.dofs_per_elem_Phi;
    if (dofs_per_elem == 1) {
        return make_numpy_array(Sol_Phi_adj_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                                mesh_.dim, py::cast(this));
    }
    const int n_elem = mesh_.GetNE();
    flux_adj_P0_.resize(num_groups_ * n_elem);
    for (int g = 0; g < num_groups_; ++g) {
        for (int e = 0; e < n_elem; ++e) {
            flux_adj_P0_(g * n_elem + e) = Sol_Phi_adj_(g * fespace_.n_Phi + e * dofs_per_elem);
        }
    }
    return make_numpy_array(flux_adj_P0_, num_groups_, mesh_.nx, mesh_.ny, mesh_.nz, 
                            mesh_.dim, py::cast(this));
}
