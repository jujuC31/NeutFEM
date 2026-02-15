/**
 * @file FEM.hpp
 * @brief Maillage cartésien et espaces éléments finis RTₖ-Pₘ (k,m = 0,1,2)
 * 
 * Ce module implémente les structures de données et fonctions de base pour
 * la méthode des éléments finis mixtes appliquée à l'équation de diffusion
 * neutronique, selon la formulation variationnelle duale de Hébert.
 * 
 * Voir la documentation complète en tête de fichier pour la théorie.
 */

#ifndef FEM_HPP
#define FEM_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <vector>
#include <array>
#include <deque>
#include <stdexcept>
#include <cmath>

// ============================================================================
// TYPES EIGEN FONDAMENTAUX
// ============================================================================

using Vec = Eigen::VectorXd;
using Vec_t = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatCol = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMat_t = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using SpMatR_t = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;

// ============================================================================
// ÉNUMÉRATIONS
// ============================================================================

/**
 * @enum FEOrder
 * @brief Ordre des éléments finis Pₖ discontinus
 */
enum class FEOrder {
    P0 = 0,     ///< Constantes par élément : 1 DOF/élément
    P1 = 1,     ///< Linéaires : 2^d DOFs/élément
    P2 = 2      ///< Quadratiques : 3^d DOFs/élément
};

/**
 * @enum RTOrder
 * @brief Ordre des éléments Raviart-Thomas
 */
enum class RTOrder {
    RT0 = 0,    ///< Normal constant par face, divergence constante
    RT1 = 1,    ///< Normal linéaire par face + bulles intérieures
    RT2 = 2     ///< Normal quadratique par face + bulles intérieures
};

// ============================================================================
// QUADRATURE DE GAUSS-LEGENDRE
// ============================================================================

/**
 * @struct GaussQuadrature1D
 * @brief Points et poids de quadrature de Gauss-Legendre 1D sur [-1,1]
 * 
 * La quadrature à n points intègre exactement les polynômes de degré ≤ 2n-1.
 */
struct GaussQuadrature1D {
    std::vector<double> points;   ///< Abscisses xᵢ ∈ [-1,1]
    std::vector<double> weights;  ///< Poids wᵢ > 0, Σwᵢ = 2
    
    /**
     * @brief Retourne la quadrature d'ordre spécifié
     * @param order Nombre de points (1-6)
     * @return Structure contenant points et poids
     */
    static GaussQuadrature1D get(int order) {
        GaussQuadrature1D q;
        switch(order) {
            case 1:
                q.points = {0.0};
                q.weights = {2.0};
                break;
            case 2:
                q.points = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
                q.weights = {1.0, 1.0};
                break;
            case 3:
                q.points = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
                q.weights = {5.0/9.0, 8.0/9.0, 5.0/9.0};
                break;
            case 4:
                q.points = {-0.861136311594053, -0.339981043584856, 
                            0.339981043584856, 0.861136311594053};
                q.weights = {0.347854845137454, 0.652145154862546, 
                             0.652145154862546, 0.347854845137454};
                break;
            case 5:
                q.points = {-0.906179845938664, -0.538469310105683, 0.0,
                            0.538469310105683, 0.906179845938664};
                q.weights = {0.236926885056189, 0.478628670499366, 0.568888888888889,
                             0.478628670499366, 0.236926885056189};
                break;
            case 6:
                q.points = {-0.932469514203152, -0.661209386466265, -0.238619186083197,
                            0.238619186083197, 0.661209386466265, 0.932469514203152};
                q.weights = {0.171324492379170, 0.360761573048139, 0.467913934572691,
                             0.467913934572691, 0.360761573048139, 0.171324492379170};
                break;
            default:
                q.points = {-0.906179845938664, -0.538469310105683, 0.0,
                            0.538469310105683, 0.906179845938664};
                q.weights = {0.236926885056189, 0.478628670499366, 0.568888888888889,
                             0.478628670499366, 0.236926885056189};
                break;
        }
        return q;
    }
};

// ============================================================================
// POLYNÔMES DE LEGENDRE
// ============================================================================

/**
 * @namespace Legendre
 * @brief Polynômes de Legendre et opérations associées sur [-1,1]
 * 
 * Les polynômes de Legendre forment une base orthogonale de L²([-1,1]) :
 *     ∫_{-1}^{1} Pₘ(x) Pₙ(x) dx = (2/(2n+1)) δₘₙ
 * 
 * Cette orthogonalité simplifie considérablement la matrice de masse.
 */
namespace Legendre {
    
    /**
     * @brief Évalue le polynôme de Legendre Pₙ(ξ)
     * 
     * Utilise la récurrence de Bonnet :
     *     (n+1)Pₙ₊₁ = (2n+1)ξPₙ - nPₙ₋₁
     * 
     * @param n  Degré du polynôme (n ≥ 0)
     * @param xi Point d'évaluation ξ ∈ [-1,1]
     * @return   Pₙ(ξ)
     */
    inline double P(int n, double xi) {
        if (n == 0) return 1.0;
        if (n == 1) return xi;
        double Pnm2 = 1.0, Pnm1 = xi, Pn = 0.0;
        for (int k = 2; k <= n; ++k) {
            Pn = ((2*k - 1) * xi * Pnm1 - (k - 1) * Pnm2) / k;
            Pnm2 = Pnm1;
            Pnm1 = Pn;
        }
        return Pn;
    }
    
    /**
     * @brief Évalue la dérivée dPₙ/dξ(ξ)
     * 
     * Utilise la formule : P'ₙ(ξ) = n(ξPₙ(ξ) - Pₙ₋₁(ξ))/(ξ² - 1)
     * 
     * Aux bords ξ = ±1, utilise la limite analytique :
     *     P'ₙ(±1) = (±1)^{n-1} · n(n+1)/2
     * 
     * @param n  Degré du polynôme (n ≥ 0)
     * @param xi Point d'évaluation ξ ∈ [-1,1]
     * @return   dPₙ/dξ(ξ)
     */
    inline double dP(int n, double xi) {
        if (n == 0) return 0.0;
        if (n == 1) return 1.0;
        
        double denom = xi * xi - 1.0;
        if (std::abs(denom) < 1e-14) {
            // Limite aux bords ξ = ±1
            double sign = (xi > 0) ? 1.0 : ((n % 2 == 0) ? 1.0 : -1.0);
            return sign * n * (n + 1) / 2.0;
        }
        return n * (xi * P(n, xi) - P(n-1, xi)) / denom;
    }
    
    /**
     * @brief Intégrale de masse ∫_{-1}^{1} Pₘ(ξ) Pₙ(ξ) dξ
     * 
     * Exploite l'orthogonalité : résultat = 2/(2n+1) si m=n, 0 sinon.
     * 
     * @param m Premier indice
     * @param n Second indice
     * @return Valeur de l'intégrale
     */
    inline double MassIntegral(int m, int n) {
        if (m != n) return 0.0;
        return 2.0 / (2.0 * n + 1.0);
    }
}

// ============================================================================
// MAILLAGE CARTÉSIEN
// ============================================================================

/**
 * @class CartesianMesh
 * @brief Maillage cartésien structuré 1D/2D/3D
 * 
 * Le maillage est défini par des "breaks" (interfaces) dans chaque direction.
 * La dimension est déduite automatiquement du nombre de breaks non-triviaux.
 * 
 * Convention de numérotation des éléments :
 *     e = iz·(nx·ny) + iy·nx + ix
 * 
 * où (ix, iy, iz) sont les indices dans chaque direction.
 * 
 * @par Exemple 2D :
 * @code
 * Vec x_breaks(4); x_breaks << 0, 1, 2, 3;  // 3 cellules en x
 * Vec y_breaks(3); y_breaks << 0, 1.5, 3;   // 2 cellules en y
 * Vec z_breaks(1); z_breaks << 0;           // 1D/2D : z trivial
 * 
 * CartesianMesh mesh(x_breaks, y_breaks, z_breaks);
 * // mesh.dim = 2, mesh.nx = 3, mesh.ny = 2, mesh.GetNE() = 6
 * @endcode
 */
class CartesianMesh {
public:
    // ========================================================================
    // DONNÉES PUBLIQUES
    // ========================================================================
    
    int dim;                ///< Dimension spatiale (1, 2 ou 3)
    int nx, ny, nz;         ///< Nombre de cellules par direction
    
    Vec_t x_breaks;         ///< Coordonnées X des interfaces (nx+1 valeurs)
    Vec_t y_breaks;         ///< Coordonnées Y des interfaces (ny+1 valeurs)
    Vec_t z_breaks;         ///< Coordonnées Z des interfaces (nz+1 valeurs)
    
    Vec_t hx, hy, hz;       ///< Tailles des cellules par direction
    Vec_t x_centers;        ///< Centres des cellules en X
    Vec_t y_centers;        ///< Centres des cellules en Y
    Vec_t z_centers;        ///< Centres des cellules en Z

    // ========================================================================
    // CONSTRUCTEUR
    // ========================================================================
    
    /**
     * @brief Construit un maillage cartésien à partir des breaks
     * 
     * @param x_brk Interfaces en X (taille ≥ 2)
     * @param y_brk Interfaces en Y (taille ≥ 1, taille 1 → 1D/2D)
     * @param z_brk Interfaces en Z (taille ≥ 1, taille 1 → 1D/2D)
     * 
     * La dimension est déduite automatiquement :
     * - dim = 1 si ny = nz = 1
     * - dim = 2 si nz = 1 et ny > 1
     * - dim = 3 si nz > 1
     */
    CartesianMesh(const Vec_t& x_brk,
                  const Vec_t& y_brk,
                  const Vec_t& z_brk);

    // ========================================================================
    // MÉTHODES D'ACCÈS
    // ========================================================================
    
    /**
     * @brief Retourne le nombre total d'éléments
     * @return nx × ny × nz
     */
    int GetNE() const;
    
    /**
     * @brief Calcule l'index global d'un élément
     * @param ix, iy, iz Indices dans chaque direction
     * @return Index global e = iz·(nx·ny) + iy·nx + ix
     */
    int ElemIndex(int ix, int iy, int iz) const;
    
    /**
     * @brief Retrouve les indices (ix, iy, iz) à partir de l'index global
     * @param e Index global de l'élément
     * @param[out] ix, iy, iz Indices par direction
     */
    void ElemCoords(int e, int& ix, int& iy, int& iz) const;
    
    /**
     * @brief Volume de l'élément e
     * @param e Index de l'élément
     * @return hx(ix) × hy(iy) × hz(iz)
     */
    double ElemVolume(int e) const;
    
    /**
     * @brief Aire d'une face de l'élément e perpendiculaire à la direction dir
     * @param e   Index de l'élément
     * @param dir Direction (0=x, 1=y, 2=z)
     * @return Aire de la face
     */
    double FaceArea(int e, int dir) const;
    
    /**
     * @brief Vecteur des volumes de tous les éléments
     * @return Vecteur de taille GetNE()
     */
    Vec_t get_vols() const;
    
    // ========================================================================
    // TRANSFORMATIONS GÉOMÉTRIQUES
    // ========================================================================
    
    /**
     * @brief Transformation physique → référence
     * 
     * Mappe un point (x,y,z) de l'élément physique e vers les coordonnées
     * de référence (ξ,η,ζ) ∈ [-1,1]³.
     * 
     * Formule : ξ = 2(x - x₀)/(x₁ - x₀) - 1
     * 
     * @param e      Index de l'élément
     * @param x,y,z  Coordonnées physiques
     * @param[out] xi, eta, zeta Coordonnées de référence
     */
    void PhysToRef(int e, double x, double y, double z, 
                   double& xi, double& eta, double& zeta) const;
    
    /**
     * @brief Transformation référence → physique
     * 
     * Mappe un point (ξ,η,ζ) ∈ [-1,1]³ vers les coordonnées physiques
     * (x,y,z) dans l'élément e.
     * 
     * Formule : x = x₀ + (1+ξ)(x₁-x₀)/2
     * 
     * @param e Index de l'élément
     * @param xi, eta, zeta Coordonnées de référence
     * @param[out] x, y, z Coordonnées physiques
     */
    void RefToPhys(int e, double xi, double eta, double zeta,
                   double& x, double& y, double& z) const;
};

// ============================================================================
// ESPACE ÉLÉMENTS FINIS RTₖ-Pₘ
// ============================================================================

/**
 * @class FESpace
 * @brief Espaces d'éléments finis Raviart-Thomas RTₖ et Pₘ discontinu
 * 
 * Cette classe gère la numérotation globale des DOFs pour le système mixte.
 * 
 * @section fespace_dofs Structure des DOFs
 * 
 * **DOFs RTₖ (courant J)** :
 * 
 * - DOFs de FACE : (k+1)^(d-1) par face, CONTINUS entre éléments voisins
 *   - Représentent les moments de J·n sur la face
 *   - Numérotés par face du maillage, pas par élément
 *   
 * - DOFs INTÉRIEURS (bulles) : k·(k+1)^(d-1) par élément par direction
 *   - Représentent les modes internes avec J·n = 0 sur les faces
 *   - DISCONTINUS, numérotés par élément
 * 
 * **DOFs Pₘ (flux φ)** :
 * 
 * - (m+1)^d DOFs par élément, complètement DISCONTINUS
 * - Produit tensoriel de Legendre : Pᵢ(ξ)·Pⱼ(η)·Pₖ(ζ)
 * 
 * @section fespace_numbering Schéma de numérotation
 * 
 * Les indices globaux sont organisés ainsi :
 * 
 * ```
 * [DOFs J] = [Jx_faces | Jy_faces | Jz_faces | Jx_interior | Jy_interior | Jz_interior]
 * [DOFs φ] = [φ_elem0 | φ_elem1 | ... | φ_elemN-1]
 * ```
 * 
 * Cette organisation facilite l'assemblage et permet d'exploiter la structure
 * par blocs du système selle.
 */
class FESpace {
public:
    // ========================================================================
    // DONNÉES PUBLIQUES
    // ========================================================================
    
    const CartesianMesh& mesh;  ///< Référence au maillage sous-jacent
    
    RTOrder rt_order;           ///< Ordre RT (0, 1 ou 2)
    FEOrder fe_order;           ///< Ordre P (0, 1 ou 2)
    
    int n_J;                    ///< Nombre total de DOFs courant
    int n_Phi;                  ///< Nombre total de DOFs flux
    
    int n_Jx, n_Jy, n_Jz;       ///< DOFs courant par direction
    
    // Compteurs locaux
    int dofs_per_face;          ///< DOFs RT par face : (k+1)^(d-1)
    int dofs_per_elem_J_interior; ///< DOFs bulles par élément par dir : k·(k+1)^(d-1)
    int dofs_per_elem_Phi;      ///< DOFs P par élément : (m+1)^d
    
    int n_J_face;               ///< Total DOFs de face (continus)
    int n_J_interior;           ///< Total DOFs intérieurs (discontinus)
    
    // ========================================================================
    // CONSTRUCTEUR
    // ========================================================================
    
    /**
     * @brief Construit l'espace EF sur un maillage donné
     * 
     * @param mesh_ Maillage cartésien de support
     * @param rt    Ordre Raviart-Thomas (défaut RT₀)
     * @param fe    Ordre P discontinu (défaut P₀)
     * 
     * @warning Pour la stabilité inf-sup, choisir k ≥ m.
     */
    FESpace(const CartesianMesh& mesh_, RTOrder rt = RTOrder::RT0, FEOrder fe = FEOrder::P0);
    
    // ========================================================================
    // INDEXATION DES DOFs COURANT (FACES)
    // ========================================================================
    
    /**
     * @brief Index global du DOF de face Jx à la position (ix, iy, iz)
     * 
     * @param ix       Position de la face (0 à nx)
     * @param iy, iz   Indices transverses
     * @param local_dof Index local sur la face (0 à dofs_per_face-1)
     * @return Index global dans le vecteur J
     * 
     * @note ix = 0 correspond à la face gauche du domaine,
     *       ix = nx correspond à la face droite.
     */
    int JxFaceIndex(int ix, int iy, int iz, int local_dof = 0) const;
    
    /** @brief Index global du DOF de face Jy (voir JxFaceIndex) */
    int JyFaceIndex(int ix, int iy, int iz, int local_dof = 0) const;
    
    /** @brief Index global du DOF de face Jz (voir JxFaceIndex) */
    int JzFaceIndex(int ix, int iy, int iz, int local_dof = 0) const;
    
    // ========================================================================
    // INDEXATION DES DOFs COURANT (INTÉRIEURS)
    // ========================================================================
    
    /**
     * @brief Index global du DOF intérieur Jx dans l'élément elem
     * 
     * @param elem      Index global de l'élément
     * @param local_dof Index local (0 à dofs_per_elem_J_interior-1)
     * @return Index global dans le vecteur J, ou -1 si k=0
     */
    int JxInteriorIndex(int elem, int local_dof) const;
    
    /** @brief Index global du DOF intérieur Jy (voir JxInteriorIndex) */
    int JyInteriorIndex(int elem, int local_dof) const;
    
    /** @brief Index global du DOF intérieur Jz (voir JxInteriorIndex) */
    int JzInteriorIndex(int elem, int local_dof) const;
    
    // ========================================================================
    // INDEXATION DES DOFs FLUX
    // ========================================================================
    
    /**
     * @brief Index global du DOF φ à la position (ix, iy, iz)
     * 
     * @param ix, iy, iz Indices de l'élément
     * @param local_dof  Index local (0 à dofs_per_elem_Phi-1)
     * @return Index global dans le vecteur φ
     */
    int PhiIndex(int ix, int iy, int iz, int local_dof = 0) const;
    
    /**
     * @brief Index global du DOF φ pour l'élément e
     * 
     * @param elem      Index global de l'élément
     * @param local_dof Index local
     * @return Index global
     */
    int PhiIndexElem(int elem, int local_dof = 0) const;
    
    // ========================================================================
    // ACCESSEURS
    // ========================================================================
    
    int GetNumFaceDofs() const { return dofs_per_face; }
    int GetNumInteriorJDofs() const { return dofs_per_elem_J_interior; }
    int GetNumPhiDofs() const { return dofs_per_elem_Phi; }
    
    /**
     * @brief Nombre total de DOFs J locaux par élément
     * @return dim × (2×dofs_per_face + dofs_per_elem_J_interior)
     */
    int GetNumLocalJDofs() const;
    
private:
    void ComputeDofCounts();
    
    // Offsets pour les différentes catégories
    int jx_face_offset_;
    int jy_face_offset_;
    int jz_face_offset_;
    int j_interior_offset_;
};

// ============================================================================
// FONCTIONS DE BASE RTₖ EN GÉOMÉTRIE CARTÉSIENNE
// ============================================================================

/**
 * @class RTBasisFunctions
 * @brief Fonctions de base RTₖ sur l'élément de référence [-1,1]^d
 * 
 * Implémente les fonctions de base Raviart-Thomas selon la structure de Hébert.
 * 
 * @section rt_face Fonctions de FACE (continues en J·n)
 * 
 * Pour la direction x, sur la face à ξ = ±1 :
 * 
 *     ψ_L^{ij}(ξ,η,ζ) = [(1-ξ)/2] · Pᵢ(η) · Pⱼ(ζ) · eₓ   (face gauche)
 *     ψ_R^{ij}(ξ,η,ζ) = [(1+ξ)/2] · Pᵢ(η) · Pⱼ(ζ) · eₓ   (face droite)
 * 
 * avec i,j = 0..k, donnant (k+1)^(d-1) fonctions par face.
 * 
 * @section rt_bubble Fonctions BULLES (intérieures, J·n = 0 sur les faces)
 * 
 *     ψ_B^{lij}(ξ,η,ζ) = (1-ξ²) · Pₗ(ξ) · Pᵢ(η) · Pⱼ(ζ) · eₓ
 * 
 * avec l = 0..k-1, i,j = 0..k, donnant k·(k+1)^(d-1) fonctions par direction.
 * 
 * Le facteur (1-ξ²) garantit ψ·eₓ = 0 pour ξ = ±1.
 * 
 * @section rt_div Divergences
 * 
 *     ∇·ψ_L = -1/2 · Pᵢ(η) · Pⱼ(ζ)   (constant en ξ)
 *     ∇·ψ_R = +1/2 · Pᵢ(η) · Pⱼ(ζ)
 *     ∇·ψ_B = [-2ξ·Pₗ(ξ) + (1-ξ²)·P'ₗ(ξ)] · Pᵢ(η) · Pⱼ(ζ)
 * 
 * Ces divergences sont en coordonnées de RÉFÉRENCE. Pour obtenir la
 * divergence physique, multiplier par 2/h (Jacobien de la transformation).
 */
class RTBasisFunctions {
public:
    /**
     * @brief Constructeur
     * @param order Ordre RT (RT0, RT1, RT2)
     * @param dim   Dimension spatiale (1, 2, 3)
     */
    RTBasisFunctions(RTOrder order, int dim);
    
    /// Nombre de fonctions de base par face : (k+1)^(d-1)
    int NumFaceBasis() const { return n_face_basis_; }
    
    /// Nombre de fonctions bulles par direction : k·(k+1)^(d-1)
    int NumInteriorBasis() const { return n_interior_basis_; }
    
    /// DOFs locaux par direction : 2·face + intérieur
    int NumLocalJDofsPerDir() const { return 2 * n_face_basis_ + n_interior_basis_; }
    
    // ========================================================================
    // ÉVALUATION DES FONCTIONS DE FACE
    // ========================================================================
    
    /**
     * @brief Évalue la fonction de base RT pour Jₓ sur une face
     * 
     * @param is_right   true pour face droite (ξ=+1), false pour gauche
     * @param local_idx  Index local 0..(k+1)^(d-1)-1
     * @param xi, eta, zeta Coordonnées de référence
     * @return Composante x de la fonction de base
     */
    double EvalJxFace(bool is_right, int local_idx, double xi, double eta, double zeta) const;
    double EvalJyFace(bool is_top, int local_idx, double xi, double eta, double zeta) const;
    double EvalJzFace(bool is_front, int local_idx, double xi, double eta, double zeta) const;
    
    // ========================================================================
    // ÉVALUATION DES FONCTIONS BULLES
    // ========================================================================
    
    /**
     * @brief Évalue la fonction bulle pour Jₓ
     * @param local_idx Index local 0..k·(k+1)^(d-1)-1
     */
    double EvalJxInterior(int local_idx, double xi, double eta, double zeta) const;
    double EvalJyInterior(int local_idx, double xi, double eta, double zeta) const;
    double EvalJzInterior(int local_idx, double xi, double eta, double zeta) const;
    
    // ========================================================================
    // ÉVALUATION DES DIVERGENCES
    // ========================================================================
    
    /**
     * @brief Divergence d'une fonction de face (coordonnées de référence)
     * @note Multiplier par 2/h pour la divergence physique
     */
    double EvalDivJxFace(bool is_right, int local_idx, double xi, double eta, double zeta) const;
    double EvalDivJyFace(bool is_top, int local_idx, double xi, double eta, double zeta) const;
    double EvalDivJzFace(bool is_front, int local_idx, double xi, double eta, double zeta) const;
    
    /** @brief Divergence d'une fonction bulle */
    double EvalDivJxInterior(int local_idx, double xi, double eta, double zeta) const;
    double EvalDivJyInterior(int local_idx, double xi, double eta, double zeta) const;
    double EvalDivJzInterior(int local_idx, double xi, double eta, double zeta) const;
    
private:
    RTOrder order_;
    int dim_;
    int k_;               ///< Ordre comme entier
    int n_face_basis_;    ///< (k+1)^(d-1)
    int n_interior_basis_;///< k·(k+1)^(d-1)
    
    void FaceIndexToTransverse(int local_idx, int& i, int& j) const;
    void InteriorIndexToMulti(int local_idx, int& l, int& i, int& j) const;
};

// ============================================================================
// FONCTIONS DE BASE Pₘ DISCONTINUES
// ============================================================================

/**
 * @class PkBasisFunctions
 * @brief Fonctions de base Pₘ discontinues (produit tensoriel de Legendre)
 * 
 * Base sur l'élément de référence [-1,1]^d :
 * 
 *     φᵢⱼₖ(ξ,η,ζ) = Pᵢ(ξ) · Pⱼ(η) · Pₖ(ζ)
 * 
 * pour i,j,k = 0..m, donnant (m+1)^d fonctions par élément.
 * 
 * @par Avantages de la base de Legendre :
 * - Orthogonalité → matrice de masse C diagonale si Σᵣ constant
 * - Bonne stabilité numérique (polynômes bornés par 1 sur [-1,1])
 * - Interprétation physique : modes constant, linéaire, quadratique...
 */
class PkBasisFunctions {
public:
    /**
     * @brief Constructeur
     * @param order Ordre P (P0, P1, P2)
     * @param dim   Dimension spatiale
     */
    PkBasisFunctions(FEOrder order, int dim);
    
    /// Nombre de fonctions de base : (m+1)^d
    int NumBasis() const { return n_basis_; }
    
    /**
     * @brief Évalue la fonction de base
     * @param local_idx Index local (0 à NumBasis()-1)
     * @param xi, eta, zeta Coordonnées de référence
     */
    double Eval(int local_idx, double xi, double eta = 0.0, double zeta = 0.0) const;
    
    /**
     * @brief Gradient de la fonction de base (coordonnées de référence)
     */
    void EvalGrad(int local_idx, double xi, double eta, double zeta,
                  double& dxi, double& deta, double& dzeta) const;
    
    /**
     * @brief Décompose l'index local en multi-index (i,j,k)
     */
    void LocalToMultiIndex(int local_idx, int& i, int& j, int& k) const;
    
    int GetOrder() const { return m_; }
    
private:
    FEOrder order_;
    int dim_;
    int m_;        ///< Ordre comme entier
    int n_basis_;  ///< (m+1)^d
};

// ============================================================================
// MATRICES D'INTÉGRATION LOCALES
// ============================================================================

/**
 * @class LocalMatrices
 * @brief Calcul des matrices élémentaires pour le système mixte
 * 
 * Calcule les matrices locales du système selle :
 * 
 *     ┌           ┐ ┌       ┐   ┌       ┐
 *     │ A_loc  Bᵀ │ │ J_loc │   │   0   │
 *     │ B_loc  C  │ │ φ_loc │ = │ f_loc │
 *     └           ┘ └       ┘   └       ┘
 * 
 * avec les intégrales :
 * 
 *     A_loc[i,j] = ∫_K (1/D) ψᵢ · ψⱼ dV      [masse RT]
 *     B_loc[i,j] = ∫_K φⱼ ∇·ψᵢ dV            [divergence]
 *     C_loc[i,j] = Σᵣ ∫_K φᵢ φⱼ dV           [réaction]
 * 
 * @section localmat_org Organisation des DOFs J locaux
 * 
 * Les DOFs sont ordonnés par direction puis par type :
 * ```
 * [Jx_left_face | Jx_right_face | Jx_interior |
 *  Jy_bottom_face | Jy_top_face | Jy_interior |
 *  Jz_back_face | Jz_front_face | Jz_interior]
 * ```
 * 
 * @section localmat_piola Transformation de Piola
 * 
 * Pour les éléments H(div), la transformation géométrique utilise Piola :
 * 
 *     J_phys = (1/|det F|) · F · J_ref
 *     ∇·J_phys = (1/|det F|) · ∇·J_ref
 * 
 * Pour un maillage cartésien, F = diag(hₓ/2, hᵧ/2, h_z/2), simplifiant :
 * 
 *     Facteur d'intégration pour A : (jac_d)² / det_J
 *     Facteur pour B : jac_d (annulé par le changement de variable)
 */
class LocalMatrices {
public:
    /**
     * @brief Constructeur
     * @param fespace Espace d'éléments finis
     * @param quadrature_order Ordre de la quadrature de Gauss (défaut 5)
     */
    LocalMatrices(const FESpace& fespace, int quadrature_order = 5);
    
    /**
     * @brief Calcule les matrices locales pour un élément
     * 
     * @param e     Index de l'élément
     * @param D     Coefficient de diffusion [cm]
     * @param Sigma Section efficace de réaction [cm⁻¹]
     */
    void Compute(int e, double D, double Sigma);
    
    /// Accès en lecture aux matrices locales
    const Mat& GetA() const { return A_local_; }
    const Mat& GetB() const { return B_local_; }
    const Mat& GetC() const { return C_local_; }
    
    /// Dimensions locales
    int NumJDofs() const { return n_J_local_; }
    int NumPhiDofs() const { return n_Phi_local_; }
    
    /**
     * @brief Mapping indices locaux → globaux pour J
     * @param ix, iy, iz Indices de l'élément
     * @param[out] indices Vecteur des indices globaux
     */
    void GetGlobalJIndices(int ix, int iy, int iz, std::vector<int>& indices) const;
    
    /**
     * @brief Mapping indices locaux → globaux pour φ
     */
    void GetGlobalPhiIndices(int ix, int iy, int iz, std::vector<int>& indices) const;
    
private:
    const FESpace& fespace_;
    const CartesianMesh& mesh_;
    
    RTBasisFunctions rt_basis_;
    PkBasisFunctions pk_basis_;
    GaussQuadrature1D quad_;
    
    int n_J_local_;    ///< dim × (2×nf + ni)
    int n_Phi_local_;  ///< (m+1)^d
    
    Mat A_local_;      ///< n_J_local × n_J_local
    Mat B_local_;      ///< n_Phi_local × n_J_local
    Mat C_local_;      ///< n_Phi_local × n_Phi_local
    
    // Offsets dans le vecteur J local
    int jx_left_offset_, jx_right_offset_, jx_int_offset_;
    int jy_bottom_offset_, jy_top_offset_, jy_int_offset_;
    int jz_back_offset_, jz_front_offset_, jz_int_offset_;
};

#endif // FEM_HPP
