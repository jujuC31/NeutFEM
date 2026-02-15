/**
 * @file wrapper.cpp
 * @brief Bindings pybind11 pour NeutFEM - Solveur de diffusion neutronique
 * 
 * Ce fichier expose l'interface Python du solveur NeutFEM via pybind11.
 * Les docstrings détaillées permettent l'autocomplétion et l'aide en ligne
 * dans les environnements Python (Jupyter, IPython, IDEs).
 * 
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // Conversion automatique numpy <-> Eigen

#include "NeutFEM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_neutfem_eigen, m) {
    m.doc() = R"pbdoc(
        NeutFEM - Solveur de diffusion neutronique multigroupe
        ======================================================
        
        Ce module implémente la résolution de l'équation de diffusion 
        neutronique multigroupe par la méthode des éléments finis mixtes
        RTₖ-Pₘ (Raviart-Thomas / Polynômes discontinus).
        
        Équation résolue
        ----------------
        Pour chaque groupe d'énergie g :
        
            -∇·(Dᵍ∇φᵍ) + Σᵣᵍφᵍ = χᵍ Σₕ(νΣf)ʰφʰ/k + Σₕ Σₛᵍ←ʰ φʰ + Qᵍ
        
        où :
            - φᵍ : flux scalaire du groupe g [n/cm²/s]
            - Dᵍ : coefficient de diffusion [cm]
            - Σᵣᵍ : section efficace de retrait [cm⁻¹]
            - k : facteur de multiplication effectif (eigenvalue)
        
        Formulation numérique
        ---------------------
        La méthode mixte-duale de Hébert introduit le courant J = -D∇φ
        et résout le système du premier ordre :
        
            J + D∇φ = 0     (loi de Fick)
            ∇·J + Σᵣφ = S   (conservation)
        
        La discrétisation RTₖ-Pₘ sur maillage cartésien garantit :
            - Conservation locale exacte du courant
            - Continuité de J·n aux interfaces
            - Stabilité inf-sup pour k ≥ m
        
        Optimisations v3
        ----------------
        - Solveur diagonal RT0-P0 : résolution en O(n), faible mémoire
        - Accélération CMFD : convergence rapide pour grands systèmes
        - Initialisation multi-grille : réduction des itérations
        
        Exemple d'utilisation
        ---------------------
        >>> import numpy as np
        >>> from neutfem import NeutFEM, BCType, BoundaryID
        >>> 
        >>> # Maillage 1D : 10 cellules de 0 à 100 cm
        >>> x = np.linspace(0, 100, 11)
        >>> y = np.array([0.0])
        >>> z = np.array([0.0])
        >>> 
        >>> # Créer le solveur RT₀-P₀ à 2 groupes
        >>> solver = NeutFEM(order=0, ng=2, x_breaks=x, y_breaks=y, z_breaks=z)
        >>> 
        >>> # Définir les sections efficaces
        >>> D = solver.get_D()
        >>> D[:] = 1.0  # cm
        >>> 
        >>> # Conditions aux limites
        >>> solver.set_bc(BoundaryID.LEFT_1D, BCType.MIRROR)
        >>> solver.set_bc(BoundaryID.RIGHT_1D, BCType.DIRICHLET, 0.0)
        >>> 
        >>> # Résoudre avec optimisations (diagonal solver)
        >>> solver.BuildMatrices()
        >>> keff = solver.SolveKeff(use_diagonal_solver=True)
        >>> print(f"keff = {keff:.5f}")
        
        Références
        ----------
        [1] Hébert A. (1993) "Application of a dual variational formulation 
            to finite element reactor calculations", ANE 20:823-845
        [2] Hébert A. (2008) "A Raviart-Thomas-Schneider solution of the 
            diffusion equation", ANE 35:363-376
        [3] Smith K.S. (1983) "Nodal Method Storage Reduction by Non-Linear
            Iteration", Trans. ANS 44:265
    )pbdoc";
    
    // ========================================================================
    // ÉNUMÉRATIONS
    // ========================================================================
    
    py::enum_<VerbosityLevel>(m, "VerbosityLevel", 
        R"pbdoc(
        Niveau de verbosité pour les sorties console.
        
        Attributes
        ----------
        SILENT : int
            Aucune sortie (pour les batchs)
        NORMAL : int
            Progression des itérations externes
        VERBOSE : int
            Résumé détaillé
        DEBUG : int
            Détails complets (itérations internes, résidus, timing)
        )pbdoc")
        .value("SILENT", VerbosityLevel::SILENT, "Aucune sortie console")
        .value("NORMAL", VerbosityLevel::NORMAL, "Progression des itérations")
        .value("VERBOSE", VerbosityLevel::VERBOSE, "Résumé détaillé")
        .value("DEBUG", VerbosityLevel::DEBUG, "Informations de débogage complètes");
    
    py::enum_<BCType>(m, "BCType",
        R"pbdoc(
        Type de condition aux limites.
        
        Pour l'équation de diffusion neutronique, les conditions aux limites
        portent sur le flux φ ou le courant J·n.
        
        Attributes
        ----------
        DIRICHLET : int
            Flux imposé : φ = φ₀
            Typiquement φ₀ = 0 pour une frontière de vide.
        
        NEUMANN : int
            Courant imposé : J·n = q
            q = 0 correspond à une condition de symétrie parfaite.
        
        ROBIN : int
            Condition d'albédo : αJ·n + βφ = 0
            Modélise une frontière partiellement réfléchissante.
            
        MIRROR : int
            Symétrie miroir : J·n = 0
            Équivalent à Neumann homogène, optimisé numériquement.
        
        PERIODIC : int
            Condition périodique : couplage des faces opposées.
        
        Notes
        -----
        La condition de vide (φ = 0 à l'infini) est approximée par :
        - DIRICHLET avec φ₀ = 0 (première approximation)
        - ROBIN avec β/α = 0.4692 (approximation de Mark)
        )pbdoc")
        .value("DIRICHLET", BCType::DIRICHLET, "Flux imposé φ = φ₀")
        .value("NEUMANN", BCType::NEUMANN, "Courant imposé J·n = q")
        .value("ROBIN", BCType::ROBIN, "Condition d'albédo αJ·n + βφ = 0")
        .value("MIRROR", BCType::MIRROR, "Symétrie miroir J·n = 0")
        .value("PERIODIC", BCType::PERIODIC, "Condition périodique");
    
    py::enum_<BoundaryID>(m, "BoundaryID",
        R"pbdoc(
        Identifiant des frontières du domaine.
        
        La convention de nommage dépend de la dimension du problème.
        Utiliser les suffixes _1D, _2D, ou _3D selon la géométrie.
        
        Géométrie 1D (x ∈ [x₀, x₁])
        ---------------------------
        LEFT_1D  : frontière à x = x₀
        RIGHT_1D : frontière à x = x₁
        
        Géométrie 2D (x ∈ [x₀, x₁], y ∈ [y₀, y₁])
        ------------------------------------------
        LEFT_2D   : frontière à x = x₀
        RIGHT_2D  : frontière à x = x₁
        BOTTOM_2D : frontière à y = y₀
        TOP_2D    : frontière à y = y₁
        
        Géométrie 3D
        ------------
        Extension naturelle avec FRONT_3D (z = z₁) et BACK_3D (z = z₀).
        )pbdoc")
        .value("LEFT_1D", BoundaryID::LEFT_1D, "Frontière gauche 1D (x = x_min)")
        .value("RIGHT_1D", BoundaryID::RIGHT_1D, "Frontière droite 1D (x = x_max)")
        .value("LEFT_2D", BoundaryID::LEFT_2D, "Frontière gauche 2D")
        .value("RIGHT_2D", BoundaryID::RIGHT_2D, "Frontière droite 2D")
        .value("TOP_2D", BoundaryID::TOP_2D, "Frontière haute 2D (y = y_max)")
        .value("BOTTOM_2D", BoundaryID::BOTTOM_2D, "Frontière basse 2D (y = y_min)")
        .value("FRONT_3D", BoundaryID::FRONT_3D, "Frontière avant 3D (z = z_max)")
        .value("BACK_3D", BoundaryID::BACK_3D, "Frontière arrière 3D (z = z_min)")
        .value("LEFT_3D", BoundaryID::LEFT_3D, "Frontière gauche 3D")
        .value("RIGHT_3D", BoundaryID::RIGHT_3D, "Frontière droite 3D")
        .value("TOP_3D", BoundaryID::TOP_3D, "Frontière haute 3D")
        .value("BOTTOM_3D", BoundaryID::BOTTOM_3D, "Frontière basse 3D");
    
    py::enum_<LinearSolverType>(m, "LinearSolverType",
        R"pbdoc(
        Type de solveur linéaire pour le système selle.
        
        Le système mixte RTₖ-Pₘ conduit à un système selle :
        
            [A   Bᵀ] [J]   [0]
            [B   C ] [φ] = [f]
        
        résolu par complément de Schur S = C + B·A⁻¹·Bᵀ.
        
        Note: Pour RT0-P0, le solveur diagonal est beaucoup plus efficace.
        
        Solveurs directs (recommandés pour problèmes < 50k DOFs)
        --------------------------------------------------------
        DIRECT_LU : Factorisation LU avec pivotage
            Robuste, usage général, O(n³) mémoire et temps.
        
        DIRECT_LDLT : Factorisation LDLᵀ
            2× plus rapide que LU pour matrices symétriques.
            Accepte les matrices indéfinies.
        
        DIRECT_LLT : Factorisation de Cholesky
            Le plus rapide, mais UNIQUEMENT pour matrices SPD.
            Échoue si Σᵣ ≤ 0 quelque part.
        
        Solveurs itératifs (pour grands problèmes)
        ------------------------------------------
        CG : Gradient conjugué
            Optimal pour matrices SPD, convergence en O(√κ) itérations.
        
        CG_DIAG : CG + préconditionneur diagonal
            Améliore le conditionnement à coût négligeable.
        
        CG_ICHOL : CG + Cholesky incomplet
            Excellent pour problèmes de diffusion. Recommandé.
        
        BICGSTAB : Bi-CGSTAB
            Pour matrices non-symétriques (cas rare en diffusion).
        
        BICGSTAB_ILU : BiCGSTAB + factorisation ILU
            Le plus robuste pour matrices difficiles.
        
        LCG : Moindres carrés conjugués
            Pour systèmes singuliers ou sur/sous-déterminés.
        
        Recommandations
        ---------------
        - RT0-P0 : Utiliser use_diagonal_solver=True (bien plus rapide)
        - RT1/RT2 petits : DIRECT_LDLT
        - RT1/RT2 grands (>50k éléments) : CG_ICHOL
        - Problèmes mal conditionnés : BICGSTAB_ILU
        )pbdoc")
        .value("DIRECT_LU", LinearSolverType::DIRECT_LU, 
               "LU avec pivotage - robuste, usage général")
        .value("DIRECT_LLT", LinearSolverType::DIRECT_LLT, 
               "Cholesky LLᵀ - le plus rapide (SPD uniquement)")
        .value("DIRECT_LDLT", LinearSolverType::DIRECT_LDLT, 
               "LDLᵀ - symétrique indéfinie acceptée")
        .value("CG", LinearSolverType::CG, 
               "Gradient conjugué sans préconditionnement")
        .value("CG_DIAG", LinearSolverType::CG_DIAG, 
               "CG + préconditionneur diagonal")
        .value("CG_ICHOL", LinearSolverType::CG_ICHOL, 
               "CG + Cholesky incomplet (recommandé)")
        .value("BICGSTAB", LinearSolverType::BICGSTAB, 
               "Bi-CGSTAB pour matrices non-symétriques")
        .value("BICGSTAB_DIAG", LinearSolverType::BICGSTAB_DIAG, 
               "BiCGSTAB + préconditionneur diagonal")
        .value("BICGSTAB_ILU", LinearSolverType::BICGSTAB_ILU, 
               "BiCGSTAB + ILU (très robuste)")
        .value("LCG", LinearSolverType::LCG, 
               "Moindres carrés conjugués");
    
    // ========================================================================
    // CLASSE PRINCIPALE NeutFEM
    // ========================================================================
    
    py::class_<NeutFEM>(m, "NeutFEM",
        R"pbdoc(
        Solveur de diffusion neutronique multigroupe par éléments finis mixtes.
        
        Cette classe implémente la résolution de l'équation de diffusion
        neutronique sur maillage cartésien 1D/2D/3D en utilisant la
        formulation mixte-duale de Hébert avec des éléments RTₖ-Pₘ.
        
        Parameters
        ----------
        order : int
            Ordre des éléments finis (0, 1, ou 2).
            Utilise RTₖ-Pₖ avec k = order.
        ng : int
            Nombre de groupes d'énergie.
        x_breaks : ndarray
            Coordonnées X des interfaces de mailles [cm].
        y_breaks : ndarray
            Coordonnées Y des interfaces (taille 1 pour 1D).
        z_breaks : ndarray
            Coordonnées Z des interfaces (taille 1 pour 1D/2D).
        
        Optimisations (v3)
        ------------------
        - use_diagonal_solver : Solveur diagonal pour RT0-P0 (5-10× plus rapide)
        - use_cmfd : Accélération CMFD (30-50% moins d'itérations)
        
        Examples
        --------
        Problème 2D avec optimisations :
        
        >>> import numpy as np
        >>> from neutfem import NeutFEM, BCType, BoundaryID
        >>> 
        >>> # Maillage 100×100
        >>> x = np.linspace(0, 100, 101)
        >>> y = np.linspace(0, 100, 101)
        >>> solver = NeutFEM(order=0, ng=2, x_breaks=x, y_breaks=y, 
        ...                  z_breaks=np.array([0.]))
        >>> 
        >>> # Configuration des XS...
        >>> solver.BuildMatrices()
        >>> 
        >>> # Résolution avec solveur diagonal (RT0-P0)
        >>> keff = solver.SolveKeff(use_diagonal_solver=True, use_cmfd=True)
        
        Notes
        -----
        La stabilité inf-sup requiert que l'ordre RT soit ≥ l'ordre P.
        Les combinaisons recommandées sont RT₀-P₀, RT₁-P₁, RT₂-P₂.
        
        See Also
        --------
        set_bc : Définition des conditions aux limites
        BuildMatrices : Assemblage du système
        SolveKeff : Résolution du problème aux valeurs propres
        )pbdoc")
        
        // ====================================================================
        // CONSTRUCTEURS
        // ====================================================================
        
        .def(py::init<int, int, const Vec_t&, const Vec_t&, const Vec_t&>(),
             py::arg("order"),
             py::arg("ng"),
             py::arg("x_breaks"),
             py::arg("y_breaks"),
             py::arg("z_breaks"),
             R"pbdoc(
             Constructeur standard RTₖ-Pₖ.
             
             Crée un solveur avec le même ordre k pour les espaces RT et P.
             C'est le choix recommandé pour la stabilité et la convergence optimale.
             
             Parameters
             ----------
             order : int
                 Ordre des éléments (0, 1, ou 2).
             ng : int
                 Nombre de groupes d'énergie.
             x_breaks : ndarray
                 Interfaces X du maillage [cm]. Doit avoir au moins 2 éléments.
             y_breaks : ndarray
                 Interfaces Y. Utiliser [0.0] pour un problème 1D.
             z_breaks : ndarray
                 Interfaces Z. Utiliser [0.0] pour un problème 1D ou 2D.
             
             Raises
             ------
             ValueError
                 Si order n'est pas dans {0, 1, 2} ou si les breaks sont invalides.
             )pbdoc")
        
        .def(py::init<int, int, int, const Vec_t&, const Vec_t&, const Vec_t&>(),
             py::arg("rt_order"),
             py::arg("p_order"),
             py::arg("ng"),
             py::arg("x_breaks"),
             py::arg("y_breaks"),
             py::arg("z_breaks"),
             R"pbdoc(
             Constructeur avec ordres distincts RTₖ-Pₘ.
             
             Permet de choisir des ordres différents pour RT et P.
             Utile pour sur-approximation (k > m) ou tests de convergence.
             
             Parameters
             ----------
             rt_order : int
                 Ordre de l'espace Raviart-Thomas (0, 1, ou 2).
             p_order : int
                 Ordre de l'espace P discontinu (0, 1, ou 2).
             ng : int
                 Nombre de groupes d'énergie.
             x_breaks, y_breaks, z_breaks : ndarray
                 Interfaces du maillage.
             
             Warnings
             --------
             Pour la stabilité inf-sup, on doit avoir rt_order >= p_order.
             Les combinaisons instables (rt_order < p_order) peuvent diverger.
             )pbdoc")
        
        // ====================================================================
        // CONFIGURATION
        // ====================================================================
        
        .def("set_bc", &NeutFEM::SetBC,
             py::arg("attr"), py::arg("type"), py::arg("value") = 0.0,
             R"pbdoc(
             Définit une condition aux limites sur une frontière.
             
             Parameters
             ----------
             attr : BoundaryID
                 Identifiant de la frontière (LEFT_1D, RIGHT_2D, etc.)
             type : BCType
                 Type de condition (DIRICHLET, NEUMANN, ROBIN, MIRROR)
             value : float, optional
                 Valeur imposée. Interprétation selon le type :
                 - DIRICHLET : φ = value
                 - NEUMANN : J·n = value
                 - ROBIN : ignoré (utiliser set_robin_coefficients)
                 - MIRROR : ignoré
             
             Examples
             --------
             >>> solver.set_bc(BoundaryID.LEFT_1D, BCType.MIRROR)
             >>> solver.set_bc(BoundaryID.RIGHT_1D, BCType.DIRICHLET, 0.0)
             )pbdoc")
        
        .def("set_robin_coefficients", &NeutFEM::SetRobinCoefficients,
             py::arg("attr"), py::arg("alpha"), py::arg("beta"),
             R"pbdoc(
             Configure les coefficients de la condition Robin.
             
             La condition Robin s'écrit : α·J·n + β·φ = 0
             
             Le rapport β/α correspond à l'inverse de l'albédo effectif.
             Pour une condition de vide, l'approximation de Mark donne β/α ≈ 0.4692.
             
             Parameters
             ----------
             attr : BoundaryID
                 Frontière concernée.
             alpha : float
                 Coefficient du courant J·n.
             beta : float
                 Coefficient du flux φ.
             
             Notes
             -----
             Appeler set_bc avec BCType.ROBIN avant cette méthode.
             
             Examples
             --------
             >>> # Condition de vide (approximation de Mark)
             >>> solver.set_bc(BoundaryID.RIGHT_1D, BCType.ROBIN)
             >>> solver.set_robin_coefficients(BoundaryID.RIGHT_1D, 0.5, 0.4692)
             )pbdoc")
        
        .def("set_linear_solver", &NeutFEM::SetLinearSolver,
             py::arg("solver_type"),
             R"pbdoc(
             Sélectionne le type de solveur linéaire.
             
             Parameters
             ----------
             solver_type : LinearSolverType
                 Type de solveur (DIRECT_LU, CG_ICHOL, etc.)
             
             Notes
             -----
             Pour RT0-P0, préférer use_diagonal_solver=True dans SolveKeff.
             )pbdoc")
        
        .def("set_tol", &NeutFEM::SetTolerance,
             py::arg("tol_keff"), py::arg("tol_flux"), py::arg("tol_L2"),
             py::arg("max_outer"), py::arg("max_inner"),
             R"pbdoc(
             Configure les critères de convergence.
             
             Parameters
             ----------
             tol_keff : float
                 Tolérance sur |kⁿ⁺¹ - kⁿ|.
             tol_flux : float
                 Tolérance sur ||φⁿ⁺¹ - φⁿ|| / ||φⁿ||.
             tol_L2 : float
                 Tolérance L² pour le solveur linéaire.
             max_outer : int
                 Nombre max d'itérations de puissance.
             max_inner : int
                 Nombre max d'itérations linéaires par groupe.
             )pbdoc")
        
        .def("set_verbosity", &NeutFEM::SetVerbosity,
             py::arg("level"),
             R"pbdoc(
             Définit le niveau de verbosité.
             
             Parameters
             ----------
             level : VerbosityLevel
                 SILENT, NORMAL, VERBOSE, ou DEBUG.
             )pbdoc")
        
        .def("set_cmfd_relaxation", &NeutFEM::SetCMFDRelaxation,
             py::arg("omega"),
             R"pbdoc(
             Configure le facteur de relaxation CMFD.
             
             Parameters
             ----------
             omega : float
                 Facteur de relaxation ∈ [0.5, 1.0].
                 Valeurs plus faibles = plus stable mais convergence plus lente.
                 Défaut: 1.0.
             
             Notes
             -----
             Réduire omega (e.g. 0.7-0.8) si des oscillations apparaissent.
             )pbdoc")
        
        .def("apply_quarter_symmetry", &NeutFEM::ApplyQuarterRotationalSymmetry,
             py::arg("axis1") = 0, py::arg("axis2") = 1,
             R"pbdoc(
             Applique la symétrie quart de cœur.
             
             Configure automatiquement les conditions MIRROR sur deux axes.
             
             Parameters
             ----------
             axis1 : int
                 Premier axe de symétrie (0=X, 1=Y, 2=Z).
             axis2 : int
                 Second axe de symétrie.
             )pbdoc")
        
        .def("add_refl", &NeutFEM::AddReflector,
             py::arg("D"), py::arg("SigR"), py::arg("SigS"),
             R"pbdoc(
             Ajoute un type de réflecteur (matériau non-fissile).
             
             Parameters
             ----------
             D : ndarray, shape (ng,)
                 Coefficients de diffusion par groupe [cm].
             SigR : ndarray, shape (ng,)
                 Sections de retrait par groupe [cm⁻¹].
             SigS : ndarray, shape (ng, ng)
                 Matrice de scattering SigS[g_to, g_from] [cm⁻¹].
             
             Returns
             -------
             int
                 Identifiant du réflecteur (pour set_refl).
             )pbdoc")
        
        .def("set_refl", &NeutFEM::SetReflector,
             py::arg("refl_id"), py::arg("dimension"), py::arg("is_upper"),
             R"pbdoc(
             Active un réflecteur sur un bord du domaine.
             
             Parameters
             ----------
             refl_id : int
                 Identifiant retourné par add_refl.
             dimension : int
                 Direction (0=x, 1=y, 2=z).
             is_upper : bool
                 True pour le bord supérieur, False pour l'inférieur.
             )pbdoc")
        
        .def("clean_refl", &NeutFEM::ClearReflectors,
             R"pbdoc(
             Supprime tous les réflecteurs actifs.
             )pbdoc")
        
        // ====================================================================
        // ASSEMBLAGE ET RÉSOLUTION
        // ====================================================================
        
        .def("BuildMatrices", &NeutFEM::BuildMatrices,
             R"pbdoc(
             Assemble toutes les matrices du système.
             
             Cette méthode doit être appelée après avoir défini les sections
             efficaces et les conditions aux limites, et avant toute résolution.
             
             Le système assemblé est :
                 [A   Bᵀ] [J]   [0]
                 [B   C ] [φ] = [f]
             
             où A est la matrice de masse RT, B l'opérateur divergence,
             et C la matrice de réaction.
             
             Notes
             -----
             - Appeler cette méthode à nouveau si les sections efficaces changent.
             - Invalide les caches (diagonal, CMFD) qui seront reconstruits.
             )pbdoc")
        
        // SolveKeff avec nouvelles options d'optimisation
        .def("SolveKeff", 
             static_cast<double (NeutFEM::*)(bool, const std::vector<int>&, bool, bool)>(&NeutFEM::SolveKeff),
             py::arg("use_coarse_init") = false,
             py::arg("coarse_factors") = std::vector<int>{},
             py::arg("use_diagonal_solver") = false,
             py::arg("use_cmfd") = false,
             R"pbdoc(
             Résout le problème aux valeurs propres k-effectif.
             
             Calcule le plus grand facteur de multiplication k et le flux
             fondamental associé par la méthode des puissances accélérée.
             
             Parameters
             ----------
             use_coarse_init : bool, optional
                 Si True, initialise avec une solution sur maillage grossier.
                 Peut accélérer la convergence pour les grands maillages.
                 Défaut: False.
             
             coarse_factors : list of int, optional
                 Facteurs de réduction par direction pour l'initialisation.
                 Exemple: [4, 4, 2] réduit le maillage d'un facteur 4×4×2.
                 Défaut: [] (pas d'initialisation grossière).
             
             use_diagonal_solver : bool, optional
                 Si True, utilise le solveur diagonal pour RT0-P0.
                 TRÈS RECOMMANDÉ pour RT0-P0 : 5-10× plus rapide, 10× moins de RAM.
                 Ignoré pour RT1/RT2 (solveur standard utilisé).
                 Défaut: False.
             
             use_cmfd : bool, optional
                 Si True, active l'accélération CMFD (Coarse Mesh Finite Difference).
                 Réduit le nombre d'itérations de 30-50% pour les grands systèmes.
                 Remplace l'accélération de Chebyshev.
                 Défaut: False.
             
             Returns
             -------
             float
                 Facteur de multiplication effectif k_eff.
                 k_eff > 1 : système sur-critique
                 k_eff = 1 : système critique
                 k_eff < 1 : système sous-critique
             
             Examples
             --------
             >>> # RT0-P0 avec toutes les optimisations
             >>> keff = solver.SolveKeff(
             ...     use_coarse_init=True, 
             ...     coarse_factors=[4, 4, 1],
             ...     use_diagonal_solver=True,
             ...     use_cmfd=True
             ... )
             
             >>> # RT1-P1 avec CMFD (solveur diagonal non disponible)
             >>> keff = solver.SolveKeff(use_cmfd=True)
             
             Notes
             -----
             Le flux solution est accessible via get_flux() après résolution.
             
             Performance (RT0-P0, maillage 100×100×10) :
             - Sans optimisation : ~60s
             - Diagonal seul : ~8s
             - Diagonal + CMFD : ~4s
             - Diagonal + CMFD + coarse init : ~2s
             )pbdoc")
        
        .def("SolveAdjoint", &NeutFEM::SolveAdjoint,
             py::arg("normalize_to_direct") = true,
             py::arg("use_direct_keff") = true,
             R"pbdoc(
             Résout le problème adjoint.
             
             Le flux adjoint φ† est solution de l'équation adjointe :
                 -∇·(D∇φ†) + Σᵣφ† = (1/k†) F†φ†
             
             où F† est l'opérateur de fission transposé :
                 (F†φ†)ᵍ = νΣfᵍ Σₕ χʰ φ†ʰ
             
             Parameters
             ----------
             normalize_to_direct : bool
                 Si True, normalise tel que ⟨φ, φ†⟩ = 1.
             use_direct_keff : bool
                 Si True, utilise le k_eff du problème direct (source fixée).
                 Si False, calcule k† par itérations de puissance.
             
             Returns
             -------
             float
                 k_eff adjoint (égal au direct si correctement convergé).
             
             Notes
             -----
             Le flux adjoint est utilisé pour :
             - Calculs de sensibilité et perturbations
             - Coefficients de pondération (importance)
             - Validation de la réciprocité
             )pbdoc")
        
        .def("SolveSubcritical", &NeutFEM::SolveSubcritical,
             R"pbdoc(
             Résout le problème sous-critique avec source externe.
             
             Combine la fission (avec k < 1) et une source externe :
                 -∇·(D∇φ) + Σᵣφ = (1/k)Fφ + Q
             
             Returns
             -------
             float
                 Facteur d'amplification M = flux_avec_fission / flux_sans_fission.
             
             Notes
             -----
             La solution existe si k < 1 (système sous-critique).
             Configurer la source externe via get_SRC().
             )pbdoc")
        
        .def("SolveCoarse", &NeutFEM::SolveCoarse,
             py::arg("refine"),
             R"pbdoc(
             Résout sur maillage grossier et projette le flux.
             
             Utilisé pour l'initialisation multi-grille. Résout un problème
             RT0-P0 sur un maillage réduit et projette le résultat.
             
             Parameters
             ----------
             refine : list of int
                 Facteurs de réduction [rx, ry, rz].
                 Le maillage grossier a (nx/rx) × (ny/ry) × (nz/rz) éléments.
             
             Returns
             -------
             tuple
                 (keff_coarse, flux_projected)
             )pbdoc")
        
        // ====================================================================
        // OPTIMISATIONS
        // ====================================================================
        
        .def("build_diagonal_cache", &NeutFEM::BuildDiagonalSchurCache,
             R"pbdoc(
             Construit le cache du solveur diagonal RT0-P0.
             
             Pré-calcule l'inverse de la diagonale du complément de Schur
             pour chaque groupe. Appelé automatiquement si use_diagonal_solver=True.
             
             Notes
             -----
             Ne fait rien si l'ordre > 0 (RT1/RT2 ou P1/P2).
             Le cache est invalidé par BuildMatrices().
             )pbdoc")
        
        .def("initialize_cmfd", &NeutFEM::InitializeCMFD,
             R"pbdoc(
             Initialise les structures de données CMFD.
             
             Pré-calcule les coefficients D̃ (diffusion aux faces).
             Appelé automatiquement si use_cmfd=True.
             )pbdoc")
        
        // ====================================================================
        // EXPORT VTK
        // ====================================================================
        
        .def("ExportVTK", &NeutFEM::ExportVTK,
             py::arg("filename"),
             py::arg("export_flux") = true,
             py::arg("export_current") = true,
             py::arg("export_xs") = false,
             py::arg("export_adjoint") = false,
             R"pbdoc(
             Exporte les résultats au format VTK pour visualisation.
             
             Le fichier VTK peut être ouvert avec ParaView, VisIt, ou tout
             autre logiciel compatible VTK.
             
             Parameters
             ----------
             filename : str
                 Nom du fichier de sortie (extension .vtk ajoutée si absente).
             export_flux : bool
                 Inclure le flux scalaire φ.
             export_current : bool
                 Inclure le courant vectoriel J.
             export_xs : bool
                 Inclure les sections efficaces (D, Σᵣ, νΣf).
             export_adjoint : bool
                 Exporter le flux adjoint au lieu du direct.
             )pbdoc")
        
        .def("ExportFluxVTK", &NeutFEM::ExportFluxVTK,
             py::arg("filename"),
             py::arg("adjoint") = false,
             R"pbdoc(
             Exporte uniquement le flux au format VTK.
             
             Version allégée de ExportVTK pour un export rapide.
             )pbdoc")
        
        .def("ExportXSVTK", &NeutFEM::ExportXSVTK,
             py::arg("filename"),
             R"pbdoc(
             Exporte les sections efficaces au format VTK.
             
             Utile pour vérifier la géométrie et les propriétés matériaux.
             )pbdoc")
        
        // ====================================================================
        // ACCESSEURS SECTIONS EFFICACES (ZERO-COPY)
        // ====================================================================
        
        .def("get_D", &NeutFEM::py_get_D,
             R"pbdoc(
             Coefficient de diffusion D [cm].
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
                 Vue zero-copy sur les données internes.
                 D[e, g] = coefficient de diffusion de l'élément e, groupe g.
             
             Notes
             -----
             Les modifications sont directement reflétées dans le solveur.
             Appeler BuildMatrices() après modification.
             )pbdoc")
        
        .def("get_SRC", &NeutFEM::py_get_SRC,
             R"pbdoc(
             Source externe Q [n/cm³/s].
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
                 Source par élément et groupe.
             )pbdoc")
        
        .def("get_SigR", &NeutFEM::py_get_SigR,
             R"pbdoc(
             Section efficace de retrait Σᵣ [cm⁻¹].
             
             Le retrait inclut l'absorption et le scattering hors-groupe :
                 Σᵣᵍ = Σₐᵍ + Σₛᵍ→autres
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
             )pbdoc")
        
        .def("get_NSF", &NeutFEM::py_get_NSF,
             R"pbdoc(
             Section de production νΣf [cm⁻¹].
             
             Produit du nombre moyen de neutrons par fission ν et de la
             section de fission Σf.
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
             )pbdoc")
        
        .def("get_KSF", &NeutFEM::py_get_KSF,
             R"pbdoc(
             Section de puissance κΣf [W/cm/neutron].
             
             Produit de l'énergie libérée par fission κ et Σf.
             Utilisé pour calculer la puissance thermique.
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
             )pbdoc")
        
        .def("get_Chi", &NeutFEM::py_get_Chi,
             R"pbdoc(
             Spectre de fission χ [-].
             
             Probabilité qu'un neutron de fission naisse dans chaque groupe.
             Doit satisfaire Σᵍ χᵍ = 1.
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
             )pbdoc")
        
        .def("get_SigS", &NeutFEM::py_get_SigS,
             R"pbdoc(
             Matrice de scattering Σₛ [cm⁻¹].
             
             Returns
             -------
             ndarray, shape (n_elements, ng, ng)
                 SigS[e, g_to, g_from] = section de transfert g_from → g_to.
             
             Notes
             -----
             Convention : SigS[g_to, g_from], pas SigS[g_from, g_to].
             )pbdoc")
        
        // ====================================================================
        // ACCESSEURS SOLUTIONS
        // ====================================================================
        
        .def("get_flux", &NeutFEM::py_get_flux,
             R"pbdoc(
             Flux neutronique scalaire φ [n/cm²/s].
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
                 Flux moyen par élément et groupe après résolution.
             
             Notes
             -----
             Pour des ordres > 0, c'est le coefficient du mode constant.
             Utiliser project_flux pour les valeurs point par point.
             )pbdoc")
        
        .def("get_flux_adj", &NeutFEM::py_get_flux_adj,
             R"pbdoc(
             Flux adjoint φ† [u.a.].
             
             Returns
             -------
             ndarray, shape (n_elements, ng)
                 Flux adjoint après appel à SolveAdjoint().
             )pbdoc")
        
        // ====================================================================
        // UTILITAIRES
        // ====================================================================
        
        .def("reset_flux", &NeutFEM::ResetFlux,
             R"pbdoc(
             Réinitialise les flux à une distribution uniforme.
             
             Utile pour relancer un calcul avec des conditions initiales propres.
             )pbdoc")
        
        .def("GetNumElements", [](const NeutFEM& self) {
                 return self.GetMesh().GetNE();
             },
             R"pbdoc(
             Retourne le nombre total d'éléments du maillage.
             
             Returns
             -------
             int
                 nx × ny × nz
             )pbdoc")
        
        .def("GetNumGroups", [](const NeutFEM& self) {
                 return self.GetFESpace().n_Phi / self.GetMesh().GetNE();
             },
             R"pbdoc(
             Retourne le nombre de groupes d'énergie.
             
             Returns
             -------
             int
             )pbdoc")
        
        .def("GetDimension", [](const NeutFEM& self) {
                 return self.GetMesh().dim;
             },
             R"pbdoc(
             Retourne la dimension spatiale du problème.
             
             Returns
             -------
             int
                 1, 2, ou 3
             )pbdoc")
        
        .def("GetLastKeff", &NeutFEM::GetLastKeff,
             R"pbdoc(
             Retourne le dernier k-effectif calculé.
             
             Returns
             -------
             float
             )pbdoc")
        
        .def("GetLastKeffAdjoint", &NeutFEM::GetLastKeffAdjoint,
             R"pbdoc(
             Retourne le k-effectif du dernier calcul adjoint.
             
             Returns
             -------
             float
             )pbdoc")
        
        .def("GetSolverName", &NeutFEM::GetSolverName,
             R"pbdoc(
             Retourne le nom du solveur linéaire actif.
             
             Returns
             -------
             str
             )pbdoc")

        .def("project_flux", &NeutFEM::ProjectFluxRefined,                       
             py::arg("refine"), py::arg("adjoint") = false,
             R"pbdoc(
             Projette le flux sur un maillage raffiné.
             
             Calcule les valeurs moyennes exactes du flux polynomial sur un
             sous-maillage plus fin, en utilisant les coefficients de Legendre.
             
             Parameters
             ----------
             refine : list of int
                 Facteurs de raffinement par direction [rx, ry, rz].
             adjoint : bool, optional
                 Projeter le flux adjoint au lieu du direct.
             
             Returns
             -------
             ndarray
                 Flux sur le maillage raffiné.
             )pbdoc")
             
        .def("project_power", &NeutFEM::ProjectPowerRefined,                     
             py::arg("refine"), py::arg("adjoint") = false,
             R"pbdoc(
             Projette la puissance (κΣf·φ) sur un maillage raffiné.
             
             Similaire à project_flux mais multiplie par κΣf pour obtenir
             la distribution de puissance thermique.
             
             Parameters
             ----------
             refine : list of int
                 Facteurs de raffinement par direction.
             adjoint : bool, optional
                 Utiliser le flux adjoint.
             
             Returns
             -------
             ndarray
                 Distribution de puissance sur le maillage raffiné [W/cm³].
             )pbdoc")
        
        .def("zoom_resolved", &NeutFEM::ZoomResolved,
             py::arg("refine"), py::arg("adjoint") = false,
             R"pbdoc(
             Zoom par re-résolution sur maillage fin.
             
             Contrairement à project_flux qui interpole, cette méthode
             re-résout le problème sur un maillage raffiné avec les
             sources figées du maillage grossier.
             
             Parameters
             ----------
             refine : list of int
                 Facteurs de raffinement par direction.
             adjoint : bool, optional
                 Utiliser le flux adjoint.
             
             Returns
             -------
             ndarray
                 Flux résolu sur le maillage raffiné.
             )pbdoc");
}
