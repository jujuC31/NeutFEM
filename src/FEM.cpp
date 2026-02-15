/**
 * @file FEM.cpp
 * @brief Implémentation du maillage et des espaces EF RTₖ-Pₘ
 * 
 * Formulation mixte-duale pour l'équation de diffusion neutronique
 * basée sur Hébert (TRIVAC) et Raviart-Thomas.
 * 
 * CORRECTIONS PAR RAPPORT À LA VERSION ORIGINALE :
 * 1. Les DOFs intérieurs (bulles) sont maintenant ACTIFS pour RT₁ et RT₂
 * 2. Les fonctions de base RT sont correctement définies selon Hébert
 * 3. Les divergences sont calculées analytiquement
 * 4. La matrice de masse A utilise l'orthogonalité de Legendre
 */

#include "FEM.hpp"
#include <iostream>
#include <cmath>

// ============================================================================
// CARTESIAN MESH
// ============================================================================

CartesianMesh::CartesianMesh(const Vec_t& x_brk,
                             const Vec_t& y_brk,
                             const Vec_t& z_brk)
    : x_breaks(x_brk), y_breaks(y_brk), z_breaks(z_brk) {

    nx = x_breaks.size() - 1;
    ny = (y_breaks.size() > 1) ? y_breaks.size() - 1 : 1;
    nz = (z_breaks.size() > 1) ? z_breaks.size() - 1 : 1;

    if (nz > 1) dim = 3;
    else if (ny > 1) dim = 2;
    else dim = 1;

    // Tailles de mailles
    hx.resize(nx);
    for (int i = 0; i < nx; ++i) {
        hx(i) = x_breaks(i+1) - x_breaks(i);
    }

    hy.resize(ny);
    if (dim >= 2) {
        for (int i = 0; i < ny; ++i) {
            hy(i) = y_breaks(i+1) - y_breaks(i);
        }
    } else {
        hy(0) = 1.0;
    }

    hz.resize(nz);
    if (dim == 3) {
        for (int i = 0; i < nz; ++i) {
            hz(i) = z_breaks(i+1) - z_breaks(i);
        }
    } else {
        hz(0) = 1.0;
    }

    // Centres
    x_centers.resize(nx);
    for (int i = 0; i < nx; ++i) {
        x_centers(i) = 0.5 * (x_breaks(i) + x_breaks(i+1));
    }

    y_centers.resize(ny);
    if (dim >= 2) {
        for (int i = 0; i < ny; ++i) {
            y_centers(i) = 0.5 * (y_breaks(i) + y_breaks(i+1));
        }
    } else {
        y_centers(0) = 0.5;
    }

    z_centers.resize(nz);
    if (dim == 3) {
        for (int i = 0; i < nz; ++i) {
            z_centers(i) = 0.5 * (z_breaks(i) + z_breaks(i+1));
        }
    } else {
        z_centers(0) = 0.5;
    }
}

int CartesianMesh::GetNE() const { 
    return nx * ny * nz; 
}

int CartesianMesh::ElemIndex(int ix, int iy, int iz) const {
    return iz * nx * ny + iy * nx + ix;
}

void CartesianMesh::ElemCoords(int e, int& ix, int& iy, int& iz) const {
    iz = e / (nx * ny);
    int rem = e % (nx * ny);
    iy = rem / nx;
    ix = rem % nx;
}

double CartesianMesh::ElemVolume(int e) const {
    int ix, iy, iz;
    ElemCoords(e, ix, iy, iz);
    return hx(ix) * hy(iy) * hz(iz);
}

double CartesianMesh::FaceArea(int e, int dir) const {
    int ix, iy, iz;
    ElemCoords(e, ix, iy, iz);
    if (dir == 0) return hy(iy) * hz(iz);
    if (dir == 1) return hx(ix) * hz(iz);
    return hx(ix) * hy(iy);
}

Vec_t CartesianMesh::get_vols() const {
    Vec_t vols(GetNE());
    for (int e = 0; e < GetNE(); ++e) {
        vols(e) = ElemVolume(e);
    }
    return vols;
}

void CartesianMesh::PhysToRef(int e, double x, double y, double z, 
                              double& xi, double& eta, double& zeta) const {
    int ix, iy, iz;
    ElemCoords(e, ix, iy, iz);
    
    double x0 = x_breaks(ix), x1 = x_breaks(ix+1);
    xi = 2.0 * (x - x0) / (x1 - x0) - 1.0;
    
    if (dim >= 2) {
        double y0 = y_breaks(iy), y1 = y_breaks(iy+1);
        eta = 2.0 * (y - y0) / (y1 - y0) - 1.0;
    } else {
        eta = 0.0;
    }
    
    if (dim == 3) {
        double z0 = z_breaks(iz), z1 = z_breaks(iz+1);
        zeta = 2.0 * (z - z0) / (z1 - z0) - 1.0;
    } else {
        zeta = 0.0;
    }
}

void CartesianMesh::RefToPhys(int e, double xi, double eta, double zeta,
                              double& x, double& y, double& z) const {
    int ix, iy, iz;
    ElemCoords(e, ix, iy, iz);
    
    double x0 = x_breaks(ix), x1 = x_breaks(ix+1);
    x = x0 + 0.5 * (xi + 1.0) * (x1 - x0);
    
    if (dim >= 2) {
        double y0 = y_breaks(iy), y1 = y_breaks(iy+1);
        y = y0 + 0.5 * (eta + 1.0) * (y1 - y0);
    } else {
        y = 0.0;
    }
    
    if (dim == 3) {
        double z0 = z_breaks(iz), z1 = z_breaks(iz+1);
        z = z0 + 0.5 * (zeta + 1.0) * (z1 - z0);
    } else {
        z = 0.0;
    }
}

// ============================================================================
// FE SPACE - RTₖ-Pₘ
// ============================================================================

FESpace::FESpace(const CartesianMesh& mesh_, RTOrder rt, FEOrder fe)
    : mesh(mesh_), rt_order(rt), fe_order(fe) {
    ComputeDofCounts();
}

void FESpace::ComputeDofCounts() {
    int k = static_cast<int>(rt_order);  // Ordre RT
    int m = static_cast<int>(fe_order);  // Ordre P
    
    // ========================================================================
    // DOFs Pₘ par élément : (m+1)^d
    // ========================================================================
    if (mesh.dim == 1) {
        dofs_per_elem_Phi = m + 1;
    } else if (mesh.dim == 2) {
        dofs_per_elem_Phi = (m + 1) * (m + 1);
    } else {
        dofs_per_elem_Phi = (m + 1) * (m + 1) * (m + 1);
    }
    
    n_Phi = mesh.GetNE() * dofs_per_elem_Phi;
    
    // ========================================================================
    // DOFs RTₖ
    // ========================================================================
    
    // DOFs par face : (k+1)^(d-1)
    if (mesh.dim == 1) {
        dofs_per_face = 1;  // Scalaire par point
    } else if (mesh.dim == 2) {
        dofs_per_face = k + 1;  // k+1 DOFs par arête
    } else {
        dofs_per_face = (k + 1) * (k + 1);  // (k+1)² DOFs par face
    }
    
    // DOFs intérieurs (BULLES) par élément par direction : k * (k+1)^(d-1)
    // IMPORTANT : Ces DOFs sont ACTIFS pour k >= 1
    if (mesh.dim == 1) {
        dofs_per_elem_J_interior = k;  // k DOFs de bulle 1D
    } else if (mesh.dim == 2) {
        dofs_per_elem_J_interior = k * (k + 1);  // k*(k+1) par direction
    } else {
        dofs_per_elem_J_interior = k * (k + 1) * (k + 1);  // k*(k+1)² par direction
    }
    
    // ========================================================================
    // Comptage total des DOFs J
    // ========================================================================
    
    // DOFs de FACE (continus entre éléments, numérotés par face du maillage)
    if (mesh.dim == 1) {
        n_Jx = (mesh.nx + 1) * dofs_per_face;
        n_Jy = 0;
        n_Jz = 0;
        n_J_face = n_Jx;
    } else if (mesh.dim == 2) {
        // Faces verticales (perpendiculaires à x) : (nx+1) * ny faces
        // Faces horizontales (perpendiculaires à y) : nx * (ny+1) faces
        int n_faces_x = (mesh.nx + 1) * mesh.ny;
        int n_faces_y = mesh.nx * (mesh.ny + 1);
        n_Jx = n_faces_x * dofs_per_face;
        n_Jy = n_faces_y * dofs_per_face;
        n_Jz = 0;
        n_J_face = n_Jx + n_Jy;
    } else {
        int n_faces_x = (mesh.nx + 1) * mesh.ny * mesh.nz;
        int n_faces_y = mesh.nx * (mesh.ny + 1) * mesh.nz;
        int n_faces_z = mesh.nx * mesh.ny * (mesh.nz + 1);
        n_Jx = n_faces_x * dofs_per_face;
        n_Jy = n_faces_y * dofs_per_face;
        n_Jz = n_faces_z * dofs_per_face;
        n_J_face = n_Jx + n_Jy + n_Jz;
    }
    
    // DOFs INTÉRIEURS (discontinus, par élément)
    n_J_interior = mesh.GetNE() * mesh.dim * dofs_per_elem_J_interior;
    
    // Total
    n_J = n_J_face + n_J_interior;
    
    // ========================================================================
    // Offsets
    // ========================================================================
    jx_face_offset_ = 0;
    jy_face_offset_ = n_Jx;
    jz_face_offset_ = n_Jx + n_Jy;
    j_interior_offset_ = n_J_face;  // Les DOFs intérieurs viennent après les faces
}

int FESpace::GetNumLocalJDofs() const {
    // Par direction : 2 faces + intérieurs
    int per_dir = 2 * dofs_per_face + dofs_per_elem_J_interior;
    return mesh.dim * per_dir;
}

int FESpace::JxFaceIndex(int ix, int iy, int iz, int local_dof) const {
    // Les faces x sont indexées par (ix, iy, iz) avec ix = 0..nx
    if (mesh.dim == 1) {
        return jx_face_offset_ + ix * dofs_per_face + local_dof;
    } else if (mesh.dim == 2) {
        // Face (ix, iy) : il y a (nx+1) faces par ligne
        int face_idx = iy * (mesh.nx + 1) + ix;
        return jx_face_offset_ + face_idx * dofs_per_face + local_dof;
    } else {
        // Face (ix, iy, iz) en 3D
        int face_idx = iz * mesh.ny * (mesh.nx + 1) + iy * (mesh.nx + 1) + ix;
        return jx_face_offset_ + face_idx * dofs_per_face + local_dof;
    }
}

int FESpace::JyFaceIndex(int ix, int iy, int iz, int local_dof) const {
    if (mesh.dim < 2) return 0;
    
    if (mesh.dim == 2) {
        // Face (ix, iy) avec iy = 0..ny
        int face_idx = iy * mesh.nx + ix;
        return jy_face_offset_ + face_idx * dofs_per_face + local_dof;
    } else {
        int face_idx = iz * (mesh.ny + 1) * mesh.nx + iy * mesh.nx + ix;
        return jy_face_offset_ + face_idx * dofs_per_face + local_dof;
    }
}

int FESpace::JzFaceIndex(int ix, int iy, int iz, int local_dof) const {
    if (mesh.dim < 3) return 0;
    
    int face_idx = iz * mesh.ny * mesh.nx + iy * mesh.nx + ix;
    return jz_face_offset_ + face_idx * dofs_per_face + local_dof;
}

int FESpace::JxInteriorIndex(int elem, int local_dof) const {
    if (dofs_per_elem_J_interior == 0) return -1;
    
    // Ordre : tous les Jx_interior, puis Jy_interior, puis Jz_interior
    int dir_offset = 0;  // direction x
    int base = j_interior_offset_ + dir_offset * mesh.GetNE() * dofs_per_elem_J_interior;
    return base + elem * dofs_per_elem_J_interior + local_dof;
}

int FESpace::JyInteriorIndex(int elem, int local_dof) const {
    if (mesh.dim < 2 || dofs_per_elem_J_interior == 0) return -1;
    
    int dir_offset = 1;  // direction y
    int base = j_interior_offset_ + dir_offset * mesh.GetNE() * dofs_per_elem_J_interior;
    return base + elem * dofs_per_elem_J_interior + local_dof;
}

int FESpace::JzInteriorIndex(int elem, int local_dof) const {
    if (mesh.dim < 3 || dofs_per_elem_J_interior == 0) return -1;
    
    int dir_offset = 2;  // direction z
    int base = j_interior_offset_ + dir_offset * mesh.GetNE() * dofs_per_elem_J_interior;
    return base + elem * dofs_per_elem_J_interior + local_dof;
}

int FESpace::PhiIndex(int ix, int iy, int iz, int local_dof) const {
    int elem = mesh.ElemIndex(ix, iy, iz);
    return elem * dofs_per_elem_Phi + local_dof;
}

int FESpace::PhiIndexElem(int elem, int local_dof) const {
    return elem * dofs_per_elem_Phi + local_dof;
}

// ============================================================================
// RT BASIS FUNCTIONS
// ============================================================================

RTBasisFunctions::RTBasisFunctions(RTOrder order, int dim) 
    : order_(order), dim_(dim), k_(static_cast<int>(order)) {
    
    // Nombre de fonctions de base par face : (k+1)^(d-1)
    if (dim == 1) {
        n_face_basis_ = 1;
    } else if (dim == 2) {
        n_face_basis_ = k_ + 1;
    } else {
        n_face_basis_ = (k_ + 1) * (k_ + 1);
    }
    
    // Nombre de fonctions de base bulles par direction : k * (k+1)^(d-1)
    if (dim == 1) {
        n_interior_basis_ = k_;
    } else if (dim == 2) {
        n_interior_basis_ = k_ * (k_ + 1);
    } else {
        n_interior_basis_ = k_ * (k_ + 1) * (k_ + 1);
    }
}

void RTBasisFunctions::FaceIndexToTransverse(int local_idx, int& i, int& j) const {
    // Décompose local_idx en indices transverses selon la dimension
    if (dim_ == 1) {
        i = 0; j = 0;
    } else if (dim_ == 2) {
        // local_idx = i, avec i = 0..k
        i = local_idx;
        j = 0;
    } else {
        // local_idx = j*(k+1) + i, avec i,j = 0..k
        i = local_idx % (k_ + 1);
        j = local_idx / (k_ + 1);
    }
}

void RTBasisFunctions::InteriorIndexToMulti(int local_idx, int& l, int& i, int& j) const {
    // Décompose local_idx en indices (l, i, j) pour les bulles
    // l = 0..k-1 est l'ordre du polynôme en direction principale
    // i, j sont les ordres transverses
    
    if (dim_ == 1) {
        l = local_idx;  // l = 0..k-1
        i = 0; j = 0;
    } else if (dim_ == 2) {
        // local_idx = i*k + l, avec l = 0..k-1, i = 0..k
        l = local_idx % k_;
        i = local_idx / k_;
        j = 0;
    } else {
        // local_idx = (j*(k+1) + i)*k + l
        int trans_idx = local_idx / k_;
        l = local_idx % k_;
        i = trans_idx % (k_ + 1);
        j = trans_idx / (k_ + 1);
    }
}

// ============================================================================
// FONCTIONS DE FACE RT
// ============================================================================

double RTBasisFunctions::EvalJxFace(bool is_right, int local_idx, 
                                     double xi, double eta, double zeta) const {
    // Fonction de base pour Jx sur face gauche ou droite
    // Face gauche (ξ=-1) : ψ = (1-ξ)/2 * P_i(η) * P_j(ζ)
    // Face droite (ξ=+1) : ψ = (1+ξ)/2 * P_i(η) * P_j(ζ)
    
    int i, j;
    FaceIndexToTransverse(local_idx, i, j);
    
    // Interpolant linéaire en ξ
    double shape_xi = is_right ? 0.5 * (1.0 + xi) : 0.5 * (1.0 - xi);
    
    // Polynômes transverses
    double P_eta = (dim_ >= 2) ? Legendre::P(i, eta) : 1.0;
    double P_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    return shape_xi * P_eta * P_zeta;
}

double RTBasisFunctions::EvalJyFace(bool is_top, int local_idx,
                                     double xi, double eta, double zeta) const {
    if (dim_ < 2) return 0.0;
    
    int i, j;
    FaceIndexToTransverse(local_idx, i, j);
    
    // Interpolant en η
    double shape_eta = is_top ? 0.5 * (1.0 + eta) : 0.5 * (1.0 - eta);
    
    // Polynômes transverses (en ξ et ζ)
    double P_xi = Legendre::P(i, xi);
    double P_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    return shape_eta * P_xi * P_zeta;
}

double RTBasisFunctions::EvalJzFace(bool is_front, int local_idx,
                                     double xi, double eta, double zeta) const {
    if (dim_ < 3) return 0.0;
    
    int i, j;
    FaceIndexToTransverse(local_idx, i, j);
    
    // Interpolant en ζ
    double shape_zeta = is_front ? 0.5 * (1.0 + zeta) : 0.5 * (1.0 - zeta);
    
    // Polynômes transverses
    double P_xi = Legendre::P(i, xi);
    double P_eta = Legendre::P(j, eta);
    
    return shape_zeta * P_xi * P_eta;
}

// ============================================================================
// FONCTIONS BULLES RT (INTÉRIEURES)
// ============================================================================

double RTBasisFunctions::EvalJxInterior(int local_idx, 
                                         double xi, double eta, double zeta) const {
    if (k_ == 0) return 0.0;  // Pas de bulles pour RT0
    
    int l, i, j;
    InteriorIndexToMulti(local_idx, l, i, j);
    
    // Bulle : (1-ξ²) * P_l(ξ) * P_i(η) * P_j(ζ)
    // Le facteur (1-ξ²) assure l'annulation sur les faces ξ=±1
    double bubble = (1.0 - xi * xi);
    double P_l_xi = Legendre::P(l, xi);
    double P_i_eta = (dim_ >= 2) ? Legendre::P(i, eta) : 1.0;
    double P_j_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    return bubble * P_l_xi * P_i_eta * P_j_zeta;
}

double RTBasisFunctions::EvalJyInterior(int local_idx,
                                         double xi, double eta, double zeta) const {
    if (dim_ < 2 || k_ == 0) return 0.0;
    
    int l, i, j;
    InteriorIndexToMulti(local_idx, l, i, j);
    
    // Bulle en direction y : (1-η²) * P_l(η) * P_i(ξ) * P_j(ζ)
    double bubble = (1.0 - eta * eta);
    double P_l_eta = Legendre::P(l, eta);
    double P_i_xi = Legendre::P(i, xi);
    double P_j_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    return bubble * P_l_eta * P_i_xi * P_j_zeta;
}

double RTBasisFunctions::EvalJzInterior(int local_idx,
                                         double xi, double eta, double zeta) const {
    if (dim_ < 3 || k_ == 0) return 0.0;
    
    int l, i, j;
    InteriorIndexToMulti(local_idx, l, i, j);
    
    // Bulle en direction z : (1-ζ²) * P_l(ζ) * P_i(ξ) * P_j(η)
    double bubble = (1.0 - zeta * zeta);
    double P_l_zeta = Legendre::P(l, zeta);
    double P_i_xi = Legendre::P(i, xi);
    double P_j_eta = Legendre::P(j, eta);
    
    return bubble * P_l_zeta * P_i_xi * P_j_eta;
}

// ============================================================================
// DIVERGENCES DES FONCTIONS RT
// ============================================================================

double RTBasisFunctions::EvalDivJxFace(bool is_right, int local_idx,
                                        double /*xi*/, double eta, double zeta) const {
    // div(ψ) = dψ_x/dξ * (2/h_x) en coordonnées physiques
    // Ici on calcule dψ_x/dξ en coordonnées de référence
    // 
    // Pour ψ = (1±ξ)/2 * P_i(η) * P_j(ζ) :
    //   dψ/dξ = ±1/2 * P_i(η) * P_j(ζ)
    
    int i, j;
    FaceIndexToTransverse(local_idx, i, j);
    
    double P_eta = (dim_ >= 2) ? Legendre::P(i, eta) : 1.0;
    double P_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    double sign = is_right ? 0.5 : -0.5;
    return sign * P_eta * P_zeta;
}

double RTBasisFunctions::EvalDivJyFace(bool is_top, int local_idx,
                                        double xi, double /*eta*/, double zeta) const {
    if (dim_ < 2) return 0.0;
    
    int i, j;
    FaceIndexToTransverse(local_idx, i, j);
    
    double P_xi = Legendre::P(i, xi);
    double P_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    double sign = is_top ? 0.5 : -0.5;
    return sign * P_xi * P_zeta;
}

double RTBasisFunctions::EvalDivJzFace(bool is_front, int local_idx,
                                        double xi, double eta, double /*zeta*/) const {
    if (dim_ < 3) return 0.0;
    
    int i, j;
    FaceIndexToTransverse(local_idx, i, j);
    
    double P_xi = Legendre::P(i, xi);
    double P_eta = Legendre::P(j, eta);
    
    double sign = is_front ? 0.5 : -0.5;
    return sign * P_xi * P_eta;
}

double RTBasisFunctions::EvalDivJxInterior(int local_idx,
                                            double xi, double eta, double zeta) const {
    if (k_ == 0) return 0.0;
    
    int l, i, j;
    InteriorIndexToMulti(local_idx, l, i, j);
    
    // ψ_x = (1-ξ²) * P_l(ξ) * P_i(η) * P_j(ζ)
    // div = dψ_x/dξ = [-2ξ * P_l(ξ) + (1-ξ²) * P'_l(ξ)] * P_i(η) * P_j(ζ)
    
    double P_l = Legendre::P(l, xi);
    double dP_l = Legendre::dP(l, xi);
    double bubble = (1.0 - xi * xi);
    
    double div_xi = -2.0 * xi * P_l + bubble * dP_l;
    
    double P_i_eta = (dim_ >= 2) ? Legendre::P(i, eta) : 1.0;
    double P_j_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    return div_xi * P_i_eta * P_j_zeta;
}

double RTBasisFunctions::EvalDivJyInterior(int local_idx,
                                            double xi, double eta, double zeta) const {
    if (dim_ < 2 || k_ == 0) return 0.0;
    
    int l, i, j;
    InteriorIndexToMulti(local_idx, l, i, j);
    
    // ψ_y = (1-η²) * P_l(η) * P_i(ξ) * P_j(ζ)
    // div = dψ_y/dη = [-2η * P_l(η) + (1-η²) * P'_l(η)] * ...
    
    double P_l = Legendre::P(l, eta);
    double dP_l = Legendre::dP(l, eta);
    double bubble = (1.0 - eta * eta);
    
    double div_eta = -2.0 * eta * P_l + bubble * dP_l;
    
    double P_i_xi = Legendre::P(i, xi);
    double P_j_zeta = (dim_ == 3) ? Legendre::P(j, zeta) : 1.0;
    
    return div_eta * P_i_xi * P_j_zeta;
}

double RTBasisFunctions::EvalDivJzInterior(int local_idx,
                                            double xi, double eta, double zeta) const {
    if (dim_ < 3 || k_ == 0) return 0.0;
    
    int l, i, j;
    InteriorIndexToMulti(local_idx, l, i, j);
    
    double P_l = Legendre::P(l, zeta);
    double dP_l = Legendre::dP(l, zeta);
    double bubble = (1.0 - zeta * zeta);
    
    double div_zeta = -2.0 * zeta * P_l + bubble * dP_l;
    
    double P_i_xi = Legendre::P(i, xi);
    double P_j_eta = Legendre::P(j, eta);
    
    return div_zeta * P_i_xi * P_j_eta;
}

// ============================================================================
// Pk BASIS FUNCTIONS
// ============================================================================

PkBasisFunctions::PkBasisFunctions(FEOrder order, int dim) 
    : order_(order), dim_(dim), m_(static_cast<int>(order)) {
    
    if (dim == 1) {
        n_basis_ = m_ + 1;
    } else if (dim == 2) {
        n_basis_ = (m_ + 1) * (m_ + 1);
    } else {
        n_basis_ = (m_ + 1) * (m_ + 1) * (m_ + 1);
    }
}

void PkBasisFunctions::LocalToMultiIndex(int local_idx, int& i, int& j, int& k) const {
    int n = m_ + 1;
    
    if (dim_ == 1) {
        i = local_idx;
        j = 0;
        k = 0;
    } else if (dim_ == 2) {
        i = local_idx % n;
        j = local_idx / n;
        k = 0;
    } else {
        i = local_idx % n;
        j = (local_idx / n) % n;
        k = local_idx / (n * n);
    }
}

double PkBasisFunctions::Eval(int local_idx, double xi, double eta, double zeta) const {
    int i, j, k;
    LocalToMultiIndex(local_idx, i, j, k);
    
    double val = Legendre::P(i, xi);
    
    if (dim_ >= 2) {
        val *= Legendre::P(j, eta);
    }
    
    if (dim_ == 3) {
        val *= Legendre::P(k, zeta);
    }
    
    return val;
}

void PkBasisFunctions::EvalGrad(int local_idx, double xi, double eta, double zeta,
                                double& dxi, double& deta, double& dzeta) const {
    int i, j, k;
    LocalToMultiIndex(local_idx, i, j, k);
    
    double Pi = Legendre::P(i, xi);
    double dPi = Legendre::dP(i, xi);
    
    if (dim_ == 1) {
        dxi = dPi;
        deta = 0.0;
        dzeta = 0.0;
    } else if (dim_ == 2) {
        double Pj = Legendre::P(j, eta);
        double dPj = Legendre::dP(j, eta);
        
        dxi = dPi * Pj;
        deta = Pi * dPj;
        dzeta = 0.0;
    } else {
        double Pj = Legendre::P(j, eta);
        double dPj = Legendre::dP(j, eta);
        double Pk = Legendre::P(k, zeta);
        double dPk = Legendre::dP(k, zeta);
        
        dxi = dPi * Pj * Pk;
        deta = Pi * dPj * Pk;
        dzeta = Pi * Pj * dPk;
    }
}

// ============================================================================
// LOCAL MATRICES
// ============================================================================

LocalMatrices::LocalMatrices(const FESpace& fespace, int quadrature_order)
    : fespace_(fespace)
    , mesh_(fespace.mesh)
    , rt_basis_(fespace.rt_order, fespace.mesh.dim)
    , pk_basis_(fespace.fe_order, fespace.mesh.dim)
    , quad_(GaussQuadrature1D::get(quadrature_order)) {
    
    // Nombre de DOFs locaux pour J
    // Par direction : 2 faces × dofs_per_face + dofs_per_elem_J_interior
    int n_per_dir = 2 * fespace_.dofs_per_face + fespace_.dofs_per_elem_J_interior;
    n_J_local_ = mesh_.dim * n_per_dir;
    n_Phi_local_ = fespace_.dofs_per_elem_Phi;
    
    A_local_.resize(n_J_local_, n_J_local_);
    B_local_.resize(n_Phi_local_, n_J_local_);
    C_local_.resize(n_Phi_local_, n_Phi_local_);
    
    // Calcul des offsets dans le vecteur J local
    int nf = fespace_.dofs_per_face;
    int ni = fespace_.dofs_per_elem_J_interior;
    
    jx_left_offset_ = 0;
    jx_right_offset_ = nf;
    jx_int_offset_ = 2 * nf;
    
    if (mesh_.dim >= 2) {
        int x_total = 2 * nf + ni;
        jy_bottom_offset_ = x_total;
        jy_top_offset_ = x_total + nf;
        jy_int_offset_ = x_total + 2 * nf;
    }
    
    if (mesh_.dim == 3) {
        int xy_total = 2 * (2 * nf + ni);
        jz_back_offset_ = xy_total;
        jz_front_offset_ = xy_total + nf;
        jz_int_offset_ = xy_total + 2 * nf;
    }
}

void LocalMatrices::Compute(int e, double D, double Sigma) {
    A_local_.setZero();
    B_local_.setZero();
    C_local_.setZero();
    
    int ix, iy, iz;
    mesh_.ElemCoords(e, ix, iy, iz);
    
    double hx = mesh_.hx(ix);
    double hy = mesh_.hy(iy);
    double hz = mesh_.hz(iz);
    
    // Jacobien de la transformation : J = diag(hx/2, hy/2, hz/2)
    double jac_x = hx / 2.0;
    double jac_y = hy / 2.0;
    double jac_z = hz / 2.0;
    double det_J = jac_x * jac_y * jac_z;
    double invD = 1.0 / D;
    
    int nq = static_cast<int>(quad_.points.size());
    int nf = fespace_.dofs_per_face;
    int ni = fespace_.dofs_per_elem_J_interior;
    
    // Quadrature tensorielle
    int ny_loop = (mesh_.dim >= 2) ? nq : 1;
    int nz_loop = (mesh_.dim == 3) ? nq : 1;
    
    for (int qx = 0; qx < nq; ++qx) {
        double xi = quad_.points[qx];
        double wx = quad_.weights[qx];
        
        for (int qy = 0; qy < ny_loop; ++qy) {
            double eta = (mesh_.dim >= 2) ? quad_.points[qy] : 0.0;
            double wy = (mesh_.dim >= 2) ? quad_.weights[qy] : 1.0;
            
            for (int qz = 0; qz < nz_loop; ++qz) {
                double zeta = (mesh_.dim == 3) ? quad_.points[qz] : 0.0;
                double wz = (mesh_.dim == 3) ? quad_.weights[qz] : 1.0;
                
                double weight = wx * wy * wz; // On applique le det_J correct ci-dessous
                double w_base = wx * wy * wz;
                
                // CORRECTION: Calcul du Jacobien et des facteurs selon la dimension
                double det_J = 0.0;
                double factor_x = 0.0, factor_y = 0.0, factor_z = 0.0;
                
                if (mesh_.dim == 1) {
                    det_J = jac_x;
                    weight *= det_J;
                    factor_x = hx / 2.0;
                } 
                else if (mesh_.dim == 2) {
                    det_J = jac_x * jac_y;
                    weight *= det_J;
                    // Facteurs de Piola pour 2D : h_transverse / h_longitudinal
                    factor_x = hy / hx;
                    factor_y = hx / hy;
                } 
                else {
                    det_J = jac_x * jac_y * jac_z;
                    weight *= det_J;
                    // Formules 3D originales
                    factor_x = 2.0 * hx / (hy * hz);
                    factor_y = 2.0 * hy / (hx * hz);
                    factor_z = 2.0 * hz / (hx * hy);
                }
                
                // ================================================================
                // Évaluer toutes les fonctions RT et leurs divergences
                // ================================================================
                std::vector<double> J_vals(n_J_local_, 0.0);
                std::vector<double> div_vals(n_J_local_, 0.0);
                
                // Direction X
                // Piola contravariant sur maillage cartésien :
                //   ψ_phys = (jac_d / det_J) * ψ_ref · e_d
                //   div(ψ_phys) = (1/det_J) * dψ_ref/dξ
                //
                // Intégrale B : ∫_K φ div(ψ) dV 
                //   = ∫_ref φ · (1/det_J) · dψ/dξ · det_J dξdηdζ
                //   = ∫_ref φ · dψ/dξ dξdηdζ
                //
                // Donc div_vals = dψ/dξ (divergence de référence SANS Jacobien)
                // et l'intégrale B utilise w_base (poids sans det_J)
                for (int f = 0; f < nf; ++f) {
                    // Face gauche
                    J_vals[jx_left_offset_ + f] = rt_basis_.EvalJxFace(false, f, xi, eta, zeta);
                    div_vals[jx_left_offset_ + f] = rt_basis_.EvalDivJxFace(false, f, xi, eta, zeta);
                    // Face droite
                    J_vals[jx_right_offset_ + f] = rt_basis_.EvalJxFace(true, f, xi, eta, zeta);
                    div_vals[jx_right_offset_ + f] = rt_basis_.EvalDivJxFace(true, f, xi, eta, zeta);
                }
                for (int b = 0; b < ni; ++b) {
                    J_vals[jx_int_offset_ + b] = rt_basis_.EvalJxInterior(b, xi, eta, zeta);
                    div_vals[jx_int_offset_ + b] = rt_basis_.EvalDivJxInterior(b, xi, eta, zeta);
                }
                
                // Direction Y
                if (mesh_.dim >= 2) {
                    for (int f = 0; f < nf; ++f) {
                        J_vals[jy_bottom_offset_ + f] = rt_basis_.EvalJyFace(false, f, xi, eta, zeta);
                        div_vals[jy_bottom_offset_ + f] = rt_basis_.EvalDivJyFace(false, f, xi, eta, zeta);
                        J_vals[jy_top_offset_ + f] = rt_basis_.EvalJyFace(true, f, xi, eta, zeta);
                        div_vals[jy_top_offset_ + f] = rt_basis_.EvalDivJyFace(true, f, xi, eta, zeta);
                    }
                    for (int b = 0; b < ni; ++b) {
                        J_vals[jy_int_offset_ + b] = rt_basis_.EvalJyInterior(b, xi, eta, zeta);
                        div_vals[jy_int_offset_ + b] = rt_basis_.EvalDivJyInterior(b, xi, eta, zeta);
                    }
                }
                
                // Direction Z
                if (mesh_.dim == 3) {
                    for (int f = 0; f < nf; ++f) {
                        J_vals[jz_back_offset_ + f] = rt_basis_.EvalJzFace(false, f, xi, eta, zeta);
                        div_vals[jz_back_offset_ + f] = rt_basis_.EvalDivJzFace(false, f, xi, eta, zeta);
                        J_vals[jz_front_offset_ + f] = rt_basis_.EvalJzFace(true, f, xi, eta, zeta);
                        div_vals[jz_front_offset_ + f] = rt_basis_.EvalDivJzFace(true, f, xi, eta, zeta);
                    }
                    for (int b = 0; b < ni; ++b) {
                        J_vals[jz_int_offset_ + b] = rt_basis_.EvalJzInterior(b, xi, eta, zeta);
                        div_vals[jz_int_offset_ + b] = rt_basis_.EvalDivJzInterior(b, xi, eta, zeta);
                    }
                }
                
                // ================================================================
                // Matrice A : (1/D) ∫ ψᵢ · ψⱼ dV
                // ================================================================
                // Pour la transformation Piola contravariant :
                //   ψ_phys = (1/|J|) * J * ψ_ref
                // Le facteur de scaling pour direction d est :
                //   (jac_d / det_J)² * det_J = jac_d² / det_J
                // Ce qui donne :
                //   X: (hx/2)² / (hx*hy*hz/8) = 2*hx/(hy*hz)
                //   Y: 2*hy/(hx*hz)
                //   Z: 2*hz/(hx*hy)
                //
                // ATTENTION: weight = wx*wy*wz*det_J, donc on utilise jac²/det_J directement
                
                //double w_base = wx * wy * wz;  // Poids de quadrature SANS det_J
                
                // Direction X : facteur = 2*hx/(hy*hz)
                //double factor_x = 2.0 * hx / (hy * hz);
                int x_end = jx_int_offset_ + ni;
                for (int i = 0; i < x_end; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        double contrib = invD * J_vals[i] * J_vals[j] * w_base * factor_x;
                        A_local_(i, j) += contrib;
                        if (i != j) A_local_(j, i) += contrib;
                    }
                }
                
                // Direction Y : facteur = 2*hy/(hx*hz)
                if (mesh_.dim >= 2) {
                    //double factor_y = 2.0 * hy / (hx * hz);
                    int y_end = jy_int_offset_ + ni;
                    for (int i = jy_bottom_offset_; i < y_end; ++i) {
                        for (int j = jy_bottom_offset_; j <= i; ++j) {
                            double contrib = invD * J_vals[i] * J_vals[j] * w_base * factor_y;
                            A_local_(i, j) += contrib;
                            if (i != j) A_local_(j, i) += contrib;
                        }
                    }
                }
                
                // Direction Z : facteur = 2*hz/(hx*hy)
                if (mesh_.dim == 3) {
                    //double factor_z = 2.0 * hz / (hx * hy);
                    int z_end = jz_int_offset_ + ni;
                    for (int i = jz_back_offset_; i < z_end; ++i) {
                        for (int j = jz_back_offset_; j <= i; ++j) {
                            double contrib = invD * J_vals[i] * J_vals[j] * w_base * factor_z;
                            A_local_(i, j) += contrib;
                            if (i != j) A_local_(j, i) += contrib;
                        }
                    }
                }
                
                // ================================================================
                // Matrice B : ∫ φⱼ div(ψᵢ) dV
                // ================================================================
                // Avec div_vals déjà multiplié par jac_d, il suffit de multiplier par w_base
                for (int phi_idx = 0; phi_idx < n_Phi_local_; ++phi_idx) {
                    double phi_val = pk_basis_.Eval(phi_idx, xi, eta, zeta);
                    
                    for (int j_idx = 0; j_idx < n_J_local_; ++j_idx) {
                        B_local_(phi_idx, j_idx) += phi_val * div_vals[j_idx] * w_base;
                    }
                }
                
                // ================================================================
                // Matrice C : Σᵣ ∫ φᵢ φⱼ dV
                // ================================================================
                for (int i = 0; i < n_Phi_local_; ++i) {
                    double phi_i = pk_basis_.Eval(i, xi, eta, zeta);
                    for (int j = 0; j <= i; ++j) {
                        double phi_j = pk_basis_.Eval(j, xi, eta, zeta);
                        double contrib = Sigma * phi_i * phi_j * weight;
                        C_local_(i, j) += contrib;
                        if (i != j) C_local_(j, i) += contrib;
                    }
                }
            }
        }
    }
}

void LocalMatrices::GetGlobalJIndices(int ix, int iy, int iz, std::vector<int>& indices) const {
    indices.clear();
    indices.reserve(n_J_local_);
    
    int nf = fespace_.dofs_per_face;
    int ni = fespace_.dofs_per_elem_J_interior;
    int elem = mesh_.ElemIndex(ix, iy, iz);
    
    // Direction X : faces + intérieurs
    for (int f = 0; f < nf; ++f) {
        indices.push_back(fespace_.JxFaceIndex(ix, iy, iz, f));      // Face gauche
    }
    for (int f = 0; f < nf; ++f) {
        indices.push_back(fespace_.JxFaceIndex(ix + 1, iy, iz, f));  // Face droite
    }
    for (int b = 0; b < ni; ++b) {
        indices.push_back(fespace_.JxInteriorIndex(elem, b));
    }
    
    // Direction Y
    if (mesh_.dim >= 2) {
        for (int f = 0; f < nf; ++f) {
            indices.push_back(fespace_.JyFaceIndex(ix, iy, iz, f));      // Face bas
        }
        for (int f = 0; f < nf; ++f) {
            indices.push_back(fespace_.JyFaceIndex(ix, iy + 1, iz, f));  // Face haut
        }
        for (int b = 0; b < ni; ++b) {
            indices.push_back(fespace_.JyInteriorIndex(elem, b));
        }
    }
    
    // Direction Z
    if (mesh_.dim == 3) {
        for (int f = 0; f < nf; ++f) {
            indices.push_back(fespace_.JzFaceIndex(ix, iy, iz, f));      // Face arrière
        }
        for (int f = 0; f < nf; ++f) {
            indices.push_back(fespace_.JzFaceIndex(ix, iy, iz + 1, f));  // Face avant
        }
        for (int b = 0; b < ni; ++b) {
            indices.push_back(fespace_.JzInteriorIndex(elem, b));
        }
    }
}

void LocalMatrices::GetGlobalPhiIndices(int ix, int iy, int iz, std::vector<int>& indices) const {
    indices.clear();
    indices.reserve(n_Phi_local_);
    
    for (int i = 0; i < n_Phi_local_; ++i) {
        indices.push_back(fespace_.PhiIndex(ix, iy, iz, i));
    }
}
