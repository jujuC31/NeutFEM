/**
 * @file NeutFEM.hpp
 * @brief Solveur de diffusion neutronique multi-groupes - Formulation mixte RTₖ-Pₖ
 * 
 * @details
 * Ce fichier définit la classe NeutFEM qui implémente un solveur d'équation de 
 * diffusion neutronique en formulation variationnelle mixte utilisant les éléments 
 * de Raviart-Thomas (RTₖ) pour le courant et les polynômes de Legendre discontinus 
 * (Pₖ) pour le flux scalaire.
 * 
 * FONCTIONNALITÉS PRINCIPALES:
 * - Calcul de k-effectif (problème aux valeurs propres)
 * - Calcul du flux adjoint (sensibilités, perturbations)
 * - Résolution de problèmes sous-critiques à source externe
 * - Projection sur maillages raffinés
 * - Export VTK pour visualisation
 * 
 * OPTIMISATIONS v3:
 * - Solveur diagonal pour RT0-P0 (faible consommation mémoire)
 * - Accélération CMFD (convergence rapide)
 * - Cache de factorisation Schur
 * - Initialisation multi-grille
 * 
 */

#ifndef NEUTFEM_HPP
#define NEUTFEM_HPP

// Inclure les dépendances existantes
#include "FEM.hpp"
#include "solvers.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <string>
#include <iostream>

namespace py = pybind11;

// ============================================================================
// ÉNUMÉRATIONS SUPPLÉMENTAIRES
// ============================================================================

/**
 * @brief Types de conditions aux limites
 */
enum class BCType {
    DIRICHLET,  ///< Vacuum (φ = valeur, condition de Marshak)
    NEUMANN,    ///< Flux imposé (J·n = valeur)
    MIRROR,     ///< Réflexion (J·n = 0, condition naturelle)
    ROBIN,      ///< Mixte (α·φ + β·J·n = valeur)
    PERIODIC    ///< Périodique (couplage des faces opposées)
};

/**
 * @brief Niveau de verbosité
 */
enum class VerbosityLevel {
    SILENT = 0,     ///< Aucune sortie
    LIGHT = 1,      ///< Messages légers
    NORMAL = 2,     ///< Messages principaux
    VERBOSE = 3,    ///< Messages détaillés
    DEBUG = 4       ///< Tous les messages
};

/**
 * @brief Identifiants des frontières (pour SetBC)
 */
enum class BoundaryID {
    // 1D
    LEFT_1D = 1,
    RIGHT_1D = 2,
    
    // 2D
    LEFT_2D = 1,
    RIGHT_2D = 2,
    TOP_2D = 3,
    BOTTOM_2D = 4,
    
    // 3D
    BACK_3D = 1,
    FRONT_3D = 2,
    LEFT_3D = 3,
    RIGHT_3D = 4,
    TOP_3D = 5,
    BOTTOM_3D = 6
};

// ============================================================================
// STRUCTURES DE DONNÉES POUR LES OPTIMISATIONS
// ============================================================================

/**
 * @brief Cache pour le solveur diagonal RT0-P0
 * 
 * Pour RT0-P0, on stocke l'inverse de la diagonale du complément de Schur
 * pour chaque groupe, permettant une résolution en O(n).
 */
struct DiagonalSchurCache {
    std::vector<Vec> S_diag_inv;    ///< 1/S_ii pour chaque groupe
    bool is_valid = false;          ///< Cache valide?
    
    void Clear() {
        S_diag_inv.clear();
        is_valid = false;
    }
};

/**
 * @brief Données pour l'accélération CMFD
 * 
 * Le CMFD (Coarse Mesh Finite Difference) accélère la convergence en
 * résolvant un système volumes-finis avec des coefficients corrigés.
 */
struct CMFDData {
    // Coefficients de diffusion aux faces (D-tilde)
    std::vector<Vec> Dtilde_x;      ///< Direction X
    std::vector<Vec> Dtilde_y;      ///< Direction Y
    std::vector<Vec> Dtilde_z;      ///< Direction Z
    
    // Facteurs de correction (D-hat)
    std::vector<Vec> Dhat_x;
    std::vector<Vec> Dhat_y;
    std::vector<Vec> Dhat_z;
    
    // Matrices CMFD
    std::vector<SpMat> M_cmfd;      ///< Une par groupe
    
    // Paramètres
    double relaxation = 1.0;        ///< Facteur de relaxation [0.5, 1.0]
    bool is_initialized = false;
    
    void Clear() {
        Dtilde_x.clear(); Dtilde_y.clear(); Dtilde_z.clear();
        Dhat_x.clear(); Dhat_y.clear(); Dhat_z.clear();
        M_cmfd.clear();
        is_initialized = false;
    }
};

// ============================================================================
// CLASSE NEUTFEM - SOLVEUR PRINCIPAL
// ============================================================================

/**
 * @brief Solveur de diffusion neutronique multi-groupes
 * 
 * Cette classe implémente un solveur complet pour l'équation de diffusion
 * neutronique multi-groupes en formulation mixte RTₖ-Pₖ.
 * 
 * USAGE TYPIQUE:
 * @code
 * // Créer le solveur
 * NeutFEM solver(0, 2, x_breaks, y_breaks, z_breaks);  // RT0, 2 groupes
 * 
 * // Configurer les sections efficaces
 * solver.D_data_ = ...;
 * solver.SigR_data_ = ...;
 * solver.NSF_data_ = ...;
 * 
 * // Assembler et résoudre
 * solver.BuildMatrices();
 * double keff = solver.SolveKeff();
 * @endcode
 */
class NeutFEM {
public:
    // =========================================================================
    // CONSTRUCTEURS
    // =========================================================================
    
    /**
     * @brief Constructeur avec ordre unique RTₖ-Pₖ
     * @param order     Ordre polynomial (0, 1, ou 2)
     * @param ng        Nombre de groupes d'énergie
     * @param x_breaks  Positions des interfaces en X
     * @param y_breaks  Positions des interfaces en Y
     * @param z_breaks  Positions des interfaces en Z
     */
    NeutFEM(int order, int ng,
            const Vec_t& x_breaks,
            const Vec_t& y_breaks,
            const Vec_t& z_breaks);
    
    /**
     * @brief Constructeur avec ordres distincts RTₖ-Pₘ
     */
    NeutFEM(int rt_order, int p_order, int ng,
            const Vec_t& x_breaks,
            const Vec_t& y_breaks,
            const Vec_t& z_breaks);
    
    /// Destructeur
    ~NeutFEM() = default;
    
    // =========================================================================
    // CONFIGURATION
    // =========================================================================
    
    /// Configure le type de solveur linéaire
    void SetLinearSolver(LinearSolverType type);
    
    /// Retourne le nom du solveur courant
    std::string GetSolverName() const;
    
    /// Configure les tolérances de convergence
    void SetTolerance(double tol_keff, double tol_flux, double tol_L2,
                      int max_outer, int max_inner);
    
    /// Configure le niveau de verbosité
    void SetVerbosity(VerbosityLevel level) { verbosity_ = level; }
    
    /// Configure une condition aux limites
    void SetBC(int attr, BCType type, double value = 0.0);
    
    /// Configure les coefficients Robin
    void SetRobinCoefficients(int attr, double alpha, double beta);
    
    /// Réinitialise les flux à leur valeur par défaut
    void ResetFlux();
    
    /// Applique la symétrie quart de cœur
    void ApplyQuarterRotationalSymmetry(int axis1 = 0, int axis2 = 1);
    
    /// Applique la symétrie centrale
    void ApplyCentralSymmetry(int axis1 = 0, int axis2 = 1);
    
    /// Configure la relaxation CMFD
    void SetCMFDRelaxation(double omega) { 
        if (cmfd_data_) cmfd_data_->relaxation = omega; 
    }
    
    // =========================================================================
    // ASSEMBLAGE
    // =========================================================================
    
    /// Assemble toutes les matrices du système
    void BuildMatrices();
    
    // =========================================================================
    // RÉSOLUTION
    // =========================================================================
    
    /**
     * @brief Calcule le k-effectif (version complète avec optimisations)
     * 
     * @param use_coarse_init     Initialisation multi-grille
     * @param coarse_factors      Facteurs de réduction {rx, ry, rz}
     * @param use_diagonal_solver Utiliser le solveur diagonal RT0-P0
     * @param use_cmfd            Activer l'accélération CMFD
     * @return Le facteur de multiplication effectif
     */
    double SolveKeff(bool use_coarse_init, const std::vector<int>& coarse_factors,
                     bool use_diagonal_solver, bool use_cmfd);
    
    /**
     * @brief Calcule le k-effectif (version simplifiée, compatibilité)
     */
    double SolveKeff(bool use_coarse_init = false, 
                     const std::vector<int>& coarse_factors = {});
    
    /**
     * @brief Calcule le flux adjoint
     * @param normalize_to_direct Normaliser tel que <φ, φ†> = 1
     * @param use_direct_keff     Utiliser le k-eff direct
     * @return Le k-eff adjoint
     */
    double SolveAdjoint(bool normalize_to_direct = true, 
                        bool use_direct_keff = true);
    
    /**
     * @brief Résout un problème sous-critique à source externe
     * @return Le facteur d'amplification M
     */
    double SolveSubcritical();
    
    /**
     * @brief Résolution sur maillage grossier (initialisation multi-grille)
     * @param refine Facteurs de réduction {rx, ry, rz}
     * @return Paire (k-eff grossier, flux projeté)
     */
    std::pair<double, Vec_t> SolveCoarse(const std::vector<int>& refine);
    
    // =========================================================================
    // OPTIMISATIONS
    // =========================================================================
    
    /// Construit le cache du solveur diagonal RT0-P0
    void BuildDiagonalSchurCache();
    
    /// Initialise les structures CMFD
    void InitializeCMFD();
    
    // =========================================================================
    // PROJECTION ET POST-TRAITEMENT
    // =========================================================================
    
    /// Projette le flux sur un maillage raffiné
    py::array_t<double> ProjectFluxRefined(const std::vector<int>& refine,
                                           bool adjoint = false) const;
    
    /// Projette la puissance sur un maillage raffiné
    py::array_t<double> ProjectPowerRefined(const std::vector<int>& refine,
                                            bool adjoint = false) const;
    
    /// Zoom par re-résolution sur maillage fin
    py::array_t<double> ZoomResolved(const std::vector<int>& refine,
                                     bool adjoint = false) const;
    
    // =========================================================================
    // EXPORT
    // =========================================================================
    
    /// Export VTK complet
    void ExportVTK(const std::string& filename,
                   bool export_flux = true,
                   bool export_current = true,
                   bool export_xs = true,
                   bool export_adjoint = false);
    
    /// Export VTK flux uniquement
    void ExportFluxVTK(const std::string& filename, bool adjoint = false);
    
    /// Export VTK sections efficaces
    void ExportXSVTK(const std::string& filename);
    
    // =========================================================================
    // ACCESSEURS PYTHON
    // =========================================================================
    
    py::array_t<double> py_get_D();
    py::array_t<double> py_get_SRC();
    py::array_t<double> py_get_SigR();
    py::array_t<double> py_get_NSF();
    py::array_t<double> py_get_KSF();
    py::array_t<double> py_get_Chi();
    py::array_t<double> py_get_SigS();
    py::array_t<double> py_get_flux();
    py::array_t<double> py_get_flux_adj();
    
    /// Accès au maillage
    const CartesianMesh& GetMesh() const { return mesh_; }
    
    /// Accès à l'espace EF
    const FESpace& GetFESpace() const { return fespace_; }
    
    /// Dernier k-eff calculé
    double GetLastKeff() const { return last_keff_direct_; }
    double GetLastKeffAdjoint() const { return last_keff_adjoint_; }
    
    /// Nombre d'éléments
    int GetNumElements() const { return mesh_.GetNE(); }
    
    /// Nombre de groupes
    int GetNumGroups() const { return num_groups_; }
    
    /// Dimension
    int GetDimension() const { return mesh_.dim; }
    
    /// Calcul de l'offset pour SigS
    int GetSigSOffset(int g_from, int g_to) const {
        return (g_to * num_groups_ + g_from) * mesh_.GetNE();
    }
    
    // =========================================================================
    // DONNÉES PUBLIQUES (SECTIONS EFFICACES)
    // =========================================================================
    
    Vec_t D_data_;      ///< Coefficient de diffusion [ng × n_elem]
    Vec_t SRC_data_;    ///< Source externe [ng × n_elem]
    Vec_t SigR_data_;   ///< Section de disparition [ng × n_elem]
    Vec_t NSF_data_;    ///< νΣf [ng × n_elem]
    Vec_t KSF_data_;    ///< κΣf [ng × n_elem]
    Vec_t Chi_data_;    ///< Spectre de fission [ng × n_elem]
    Vec_t SigS_data_;   ///< Scattering [ng × ng × n_elem]
    
    // =========================================================================
    // SOLUTIONS
    // =========================================================================
    
    Vec_t Sol_Phi_;     ///< Flux scalaire [ng × n_Phi]
    Vec_t Sol_J_;       ///< Courant [ng × n_J]
    Vec_t Sol_Phi_adj_; ///< Flux adjoint [ng × n_Phi]
    Vec_t Sol_J_adj_;   ///< Courant adjoint [ng × n_J]
    
    // =========================================================================
    // STUBS POUR COMPATIBILITÉ
    // =========================================================================
    
    int AddReflector(py::array_t<double>, py::array_t<double>, py::array_t<double>);
    void SetReflector(int, int, bool);
    void ClearReflectors();
    void SelectOptimalSolver();
    
private:
    // =========================================================================
    // MÉTHODES PRIVÉES - ASSEMBLAGE
    // =========================================================================
    
    void AssembleA(int group);
    void AssembleB();
    void AssembleC(int group);
    void AssembleFissionMatrix(int group);
    void AssembleScatteringMatrix(int g_from, int g_to);
    void ApplyDirichletToA(int group);
    
    SpMat AssembleWeightedMassMatrix(const Vec_t& coeff_per_elem) const;
    
    // =========================================================================
    // MÉTHODES PRIVÉES - RÉSOLUTION
    // =========================================================================
    
    void SolveGroupInternal(int g, const Vec& source);
    void SolveGroupInternalAdjoint(int g, const Vec& source);
    void SolveDiagonalSchur(int g, const Vec& rhs, Vec& Phi_sol, Vec& J_sol);
    
    void ApplyBoundaryConditions(int group, Vec& rhs);
    
    void BuildFissionRHS(int g, const Vec& total_fiss, double keff, Vec& rhs) const;
    void BuildFissionRHSAdjoint(int g, const Vec& total_chi_adj, double keff, Vec& rhs) const;
    void BuildExternalSourceRHS(int g, Vec& rhs) const;
    
    // =========================================================================
    // MÉTHODES PRIVÉES - CMFD
    // =========================================================================
    
    void ComputeDtildeCoefficients();
    void UpdateDhatCoefficients();
    Vec ApplyCMFDCorrection(int g, const Vec& total_fiss, double keff);
    
    // =========================================================================
    // MÉTHODES PRIVÉES - UTILITAIRES
    // =========================================================================
    
    int GetBoundaryAttribute(int dim, int direction, bool is_upper) const;
    double ComputeBoundaryFaceIntegral(int local_face_dof, double face_area) const;
    
    template<typename... Args>
    void Log(VerbosityLevel level, Args&&... args) const {
        if (verbosity_ >= level) {
            (std::cout << ... << std::forward<Args>(args)) << std::endl;
        }
    }
    
    // =========================================================================
    // MEMBRES PRIVÉS
    // =========================================================================
    
    CartesianMesh mesh_;       ///< Maillage cartésien (de FEM.hpp)
    FESpace fespace_;          ///< Espace EF (de FEM.hpp)
    
    int num_groups_;
    int rt_order_int_;
    int p_order_int_;
    RTOrder rt_order_;
    FEOrder fe_order_;
    
    // Matrices globales
    std::vector<SpMat> A_mats_;     ///< Masse RT par groupe
    SpMat B_mat_;                    ///< Divergence
    SpMat BT_mat_;                   ///< Transposée de B
    std::vector<SpMat> C_mats_;     ///< Réaction par groupe
    std::vector<SpMat> M_fiss_;     ///< Fission par groupe
    std::vector<SpMat> M_scatter_;  ///< Scattering
    std::vector<SpMat> M_chi_;      ///< Spectre χ
    std::vector<SpMat> M_nsf_mass_; ///< νΣf pondéré
    
    // Solveurs (de solvers.hpp)
    std::unique_ptr<SchurSolver> schur_solver_;
    std::unique_ptr<LocalMatrices> local_matrices_;
    
    // Caches d'optimisation
    std::unique_ptr<DiagonalSchurCache> diag_schur_cache_;
    std::unique_ptr<CMFDData> cmfd_data_;
    bool schur_factorized_;
    
    // Conditions aux limites
    std::map<int, BCType> bc_types_;
    std::map<int, double> bc_values_;
    std::map<int, double> robin_alpha_;
    std::map<int, double> robin_beta_;
    
    // Configuration
    LinearSolverType linear_solver_type_;
    bool has_quarter_symmetry_;
    bool has_central_symmetry_;
    int sym_axis1_, sym_axis2_;
    
    // Tolérances
    double tol_keff_;
    double tol_flux_;
    double tol_L2_;
    int max_outer_iter_;
    int max_inner_iter_;
    
    // État
    VerbosityLevel verbosity_;
    double last_keff_direct_;
    double last_keff_adjoint_;
    bool has_valid_keff_;
    bool has_valid_adjoint_;
    
    // Buffers temporaires pour export Python
    mutable Vec_t flux_P0_;
    mutable Vec_t flux_adj_P0_;
};

// ============================================================================
// FONCTION HELPER POUR NUMPY
// ============================================================================

/**
 * @brief Crée un tableau numpy à partir d'un vecteur Eigen
 */
py::array_t<double> make_numpy_array(Vec_t& data, int ng, int nx, int ny, int nz,
                                     int dim, py::object owner);

#endif // NEUTFEM_HPP
