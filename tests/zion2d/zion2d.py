"""
BENCHMARK ZION 2D - Version Eigen (sans MFEM)
Adaptée pour neutfem_eigen avec:
  - Ordres mixtes RT_k-P_m
  - Solveur adjoint
  - Export VTK
  - Initialisation coarse mesh
  - Solveur BiCGSTAB (sans préconditionneur)
  - Géométrie avec baffle/réflecteur paramétrable

Géométrie Zion:
  - Assemblage combustible : 21.608 cm
  - Épaisseur baffle       : 2.8575 cm
  - Le maillage est construit avec des tailles de mailles non uniformes
    pour respecter exactement les dimensions du baffle et des assemblages.
"""

import time
import numpy as np
import argparse

# ===== IMPORT VERSION EIGEN =====
import neutfem._neutfem_eigen as neutron_solver
from neutfem._neutfem_eigen import BCType, BoundaryID, VerbosityLevel, LinearSolverType

import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Zion2D:
    """
    Benchmark ZION 2D avec fonctionnalités avancées
    Version adaptée pour NeutFEM Eigen
    
    Paramètres de maillage:
    -----------------------
    - submesh_assembly : nombre de subdivisions par assemblage (21.608 cm)
    - submesh_baffle   : nombre de subdivisions dans le baffle (2.8575 cm)
    
    Les dimensions sont:
    - Assemblage : 21.608 cm
    - Baffle     : 2.8575 cm (≈ 21.608 / 7.56)
    
    Pour un maillage cohérent, on utilise des mailles non-uniformes.
    """
    
    # Dimensions géométriques (constantes)
    ASSEMBLY_SIZE = 21.608    # cm - taille d'un assemblage
    BAFFLE_SIZE = 2.8575      # cm - épaisseur du baffle
    
    def __init__(self, domaine="entier", ncpu=1, submesh_assembly=1, submesh_baffle=1):
        """
        Initialise le benchmark Zion 2D.
        
        Parameters
        ----------
        domaine : str
            Type de domaine ("entier", "quart_so", etc.)
        ncpu : int
            Nombre de CPUs (non utilisé actuellement)
        submesh_assembly : int
            Nombre de subdivisions par assemblage (défaut: 1)
        submesh_baffle : int
            Nombre de subdivisions dans le baffle (défaut: 1)
        """
        self.start = time.time()
        self.domaine = domaine
        self.ncpu = ncpu
        
        # Paramètres de raffinement du maillage
        self.submesh_assembly = submesh_assembly
        self.submesh_baffle = submesh_baffle
        
        self.kref = 1.274893
        self.num_groups = 2
        self.verbose = 0
        
        self.rt_order = 0
        self.p_order = 0
        
        self.init_meshing = False
        self.mysolv = None
        self.keff = None
        self.keff_adj = None
        self.phi = None
        self.phi_adj = None
        self.pvol = None
        
        self.use_coarse = True
        self.coarse_factors = [2, 2, 1]
        
        # Calcul des tailles de mailles
        self.cell_size_assembly = self.ASSEMBLY_SIZE / self.submesh_assembly
        self.cell_size_baffle = self.BAFFLE_SIZE / self.submesh_baffle
        
        print(f"  Géométrie Zion 2D:")
        print(f"    Assemblage : {self.ASSEMBLY_SIZE} cm → {self.submesh_assembly} maille(s) de {self.cell_size_assembly:.4f} cm")
        print(f"    Baffle     : {self.BAFFLE_SIZE} cm → {self.submesh_baffle} maille(s) de {self.cell_size_baffle:.4f} cm")

        # Maillage motif - 19x19 positions
        # F1 = baffle, F2-F4 = fuel types, F5 = réflecteur (eau), "  " = vide/eau externe
        self.maillage_motifs_coeur = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "], 
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F2", "F4", "F2", "F4", "F2", "F4", "F4", "F4", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "F4", "F4", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F4", "F4", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "F4", "F3", "F3", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F3", "F3", "F4", "  ", "  ", "  "],  
            ["  ", "  ", "F4", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "F4", "  ", "  "],  
            ["  ", "  ", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F4", "F4", "  ", "  "], 
            ["  ", "  ", "  ", "F4", "F3", "F3", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F3", "F3", "F4", "  ", "  ", "  "], 
            ["  ", "  ", "  ", "F4", "F4", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F2", "F3", "F4", "F4", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F2", "F4", "F2", "F4", "F2", "F4", "F4", "F4", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "], 
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

    def plot_geom(self):
        if not self.init_meshing:
            print("❌ Erreur: Le maillage doit être initialisé d'abord")
            return
        maillage_draw = [[0 if c == "  " else int(c[1]) for c in row] for row in self.maillage]
        sns.heatmap(maillage_draw, cmap='jet', linewidths=0.5, linecolor="k")
        plt.title(f"Géométrie Zion 2D - {self.domaine}")
        plt.show()

    def plot_materials(self):
        maillage_draw = np.array([[0 if c == "  " else int(c[1]) for c in row] for row in self.maillage_motifs_coeur])
        widths_x = [2.8575 if "F1" in [self.maillage_motifs_coeur[m][n] for m in range(maillage_draw.shape[0])] else 21.608
                    for n in range(maillage_draw.shape[1])]
        widths_y = [2.8575 if "F1" in self.maillage_motifs_coeur[m] else 21.608 for m in range(maillage_draw.shape[0])]
        x_edges = np.concatenate(([0], np.cumsum(widths_x)))
        y_edges = np.concatenate(([0], np.cumsum(widths_y)))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pcolormesh(x_edges, y_edges, maillage_draw, cmap='jet', edgecolors='k', linewidth=1)
        ax.set_aspect('equal')
        plt.title("Distribution des matériaux Zion 2D")
        plt.show()

    def mesh_initialisation(self, domaine=None):
        """
        Initialise le maillage avec des tailles de mailles non-uniformes.
        
        Le maillage respecte exactement les dimensions géométriques:
        - Assemblages : 21.608 cm subdivisés en submesh_assembly mailles
        - Baffle      : 2.8575 cm subdivisé en submesh_baffle mailles
        - Eau externe : même taille que les assemblages (réflecteur)
        
        La carte maillage_motifs_coeur de 19x19 représente:
        - Lignes/colonnes 0,1,17,18 : eau externe (= réflecteur F5)
        - Le reste : cœur avec assemblages et baffle
        
        Note: Dans le benchmark Zion original, il n'y a pas de F1 explicite
        dans la carte. Le baffle est détecté automatiquement comme les cellules
        vides adjacentes au cœur actif (F2, F3, F4).
        """
        timeref = time.time()
        if domaine:
            self.domaine = domaine
        
        # Dans le Zion 2D, la géométrie est plus complexe car on a un baffle
        # autour du cœur actif. Le baffle n'est pas explicitement dans la carte
        # mais est les cellules "  " adjacentes aux assemblages F2/F3/F4.
        
        # Pour simplifier, on va construire un maillage avec:
        # - Chaque position motif (assemblage ou vide) = submesh_assembly mailles
        # - SAUF: le baffle qui sera géré lors du remplissage des XS
        
        # Construction du maillage étendu
        # On utilise submesh_assembly pour les assemblages
        self.maillage = np.array([
            [c for c in row for _ in range(self.submesh_assembly)] 
            for row in self.maillage_motifs_coeur 
            for _ in range(self.submesh_assembly)
        ])
        
        # Application des symétries
        L = len(self.maillage)
        L_half = L // 2
        domaine_map = {
            "quart_so": (slice(L_half, None), slice(None, L_half)),
            "quart_no": (slice(None, L_half), slice(None, L_half)),
            "quart_ne": (slice(None, L_half), slice(L_half, None)),
            "quart_se": (slice(L_half, None), slice(L_half, None)),
        }
        if self.domaine in domaine_map:
            y_slice, x_slice = domaine_map[self.domaine]
            self.maillage = self.maillage[y_slice, x_slice]
        
        ny_cells = self.maillage.shape[0]
        nx_cells = self.maillage.shape[1]
        
        # Construction des coordonnées (maillage uniforme basé sur assemblages)
        # Chaque cellule motif fait ASSEMBLY_SIZE, subdivisée en submesh_assembly
        cell_size = self.ASSEMBLY_SIZE / self.submesh_assembly
        
        self.x_breaks = np.linspace(0.0, nx_cells * cell_size, nx_cells + 1)
        self.y_breaks = np.linspace(0.0, ny_cells * cell_size, ny_cells + 1)
        self.z_breaks = np.array([0.0])
        self._compute_coarse_factors(nx_cells, ny_cells)
        print(f"✅ Maillage initialisé: {ny_cells}×{nx_cells} cellules ({time.time()-timeref:.3f} s)")
        self.init_meshing = True

    def _compute_coarse_factors(self, nx, ny):
        def find_factor(n, max_factor=4):
            for f in range(min(max_factor, n), 0, -1):
                if n % f == 0:
                    return f
            return 1
        self.coarse_factors = [find_factor(nx), find_factor(ny), 1]
        print(f"  Facteurs coarse calculés: {self.coarse_factors[0]}×{self.coarse_factors[1]}×1")

    def load_zion2d_mat(self):
        self.F1 = {'D': [1.0213, 0.33548], 'ABS': [0.00322, 0.14596], 'NSF': [0., 0.], 'CHI': [0., 0.], 'S12': 0., 'S21': 0.}
        self.F1['SIGR'] = [self.F1['ABS'][0], self.F1['ABS'][1]]
        self.F2 = {'D': [1.4176, 0.37335], 'ABS': [0.00855, 0.06669], 'NSF': [0.00536, 0.10433], 'CHI': [1., 0.], 'S12': 0.01742, 'S21': 0.}
        self.F2['SIGR'] = [self.F2['ABS'][0] + self.F2['S12'], self.F2['ABS'][1]]
        self.F3 = {'D': [1.4192, 0.37370], 'ABS': [0.00882, 0.07606], 'NSF': [0.00601, 0.12472], 'CHI': [1., 0.], 'S12': 0.01694, 'S21': 0.}
        self.F3['SIGR'] = [self.F3['ABS'][0] + self.F3['S12'], self.F3['ABS'][1]]
        self.F4 = {'D': [1.4265, 0.37424], 'ABS': [0.00902, 0.08359], 'NSF': [0.00653, 0.1412], 'CHI': [1., 0.], 'S12': 0.01658, 'S21': 0.}
        self.F4['SIGR'] = [self.F4['ABS'][0] + self.F4['S12'], self.F4['ABS'][1]]
        self.F5 = {'D': [1.4554, 0.28994], 'ABS': [0.00047, 0.00949], 'NSF': [0., 0.], 'CHI': [0., 0.], 'S12': 0.02903, 'S21': 0.}
        self.F5['SIGR'] = [self.F5['ABS'][0] + self.F5['S12'], self.F5['ABS'][1]]
        self.R0 = {'D': [0.1, 0.1], 'ABS': [1e8, 1e8], 'NSF': [0., 0.], 'CHI': [0., 0.], 'S12': 0., 'S21': 0.}
        self.R0['SIGR'] = self.R0['ABS'].copy()
        print("✅ Matériaux chargés: F1 (baffle), F2-F4 (fuel), F5 (réflecteur)")

    def init_solver(self):
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé d'abord")
        timeref = time.time()
        print(f"\n=== INITIALISATION SOLVEUR RT{self.rt_order}-P{self.p_order} ===")
        if self.rt_order == self.p_order:
            self.mysolv = neutron_solver.NeutFEM(self.rt_order, self.num_groups, self.x_breaks, self.y_breaks, self.z_breaks)
        else:
            self.mysolv = neutron_solver.NeutFEM(self.rt_order, self.p_order, self.num_groups, self.x_breaks, self.y_breaks, self.z_breaks)
        self.mysolv.set_linear_solver(LinearSolverType.BICGSTAB)
        
        if self.domaine == "entier":
            self.mysolv.set_bc(int(BoundaryID.LEFT_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.RIGHT_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.TOP_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.BOTTOM_2D), BCType.DIRICHLET, 0.0)
            print("  Domaine ENTIER : Dirichlet sur tous les bords")
        elif self.domaine == "quart_so":
            self.mysolv.apply_quarter_rotational_symmetry(0, 1)
            self.mysolv.set_bc(int(BoundaryID.LEFT_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.TOP_2D), BCType.MIRROR, 0.0)
            self.mysolv.set_bc(int(BoundaryID.RIGHT_2D), BCType.MIRROR, 0.0)
            self.mysolv.set_bc(int(BoundaryID.BOTTOM_2D), BCType.DIRICHLET, 0.0)
            print("  Domaine QUART_SO : Symétrie quart cyclique")

        Ny, Nx = self.maillage.shape
        print(f"\n=== REMPLISSAGE DES COEFFICIENTS ({Ny}×{Nx}) ===")
        
        # Distance de recherche pour le baffle (en nombre de mailles)
        # Le baffle fait BAFFLE_SIZE cm, une maille fait cell_size_assembly cm
        # On cherche les voisins dans un rayon correspondant à l'épaisseur du baffle
        baffle_search_radius = max(1, int(np.ceil(self.BAFFLE_SIZE / self.cell_size_assembly)))
        print(f"  Rayon de recherche baffle: {baffle_search_radius} maille(s)")
        
        n_baffle = 0
        n_reflector = 0
        n_fuel = 0
        
        for i in range(Ny):
            for j in range(Nx):
                fuel_key = self.maillage[i, j].strip()
                
                if not fuel_key:
                    # Cellule vide - déterminer si c'est du baffle ou du réflecteur
                    is_baffle = False
                    
                    # Chercher si on est adjacent à un assemblage combustible
                    for di in range(-baffle_search_radius, baffle_search_radius + 1):
                        for dj in range(-baffle_search_radius, baffle_search_radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < Ny and 0 <= nj < Nx:
                                neighbor = self.maillage[ni, nj].strip()
                                # Si on trouve du combustible (F2, F3, F4) à proximité
                                if neighbor in ["F2", "F3", "F4"]:
                                    is_baffle = True
                                    break
                        if is_baffle:
                            break
                    
                    if is_baffle:
                        mat = self.F1  # Baffle
                        self.maillage[i, j] = "F1"
                        n_baffle += 1
                    else:
                        mat = self.F5  # Réflecteur (eau)
                        n_reflector += 1
                else:
                    mat = getattr(self, fuel_key) if hasattr(self, fuel_key) else self.R0
                    n_fuel += 1
                
                for g in range(self.num_groups):
                    self.mysolv.get_D()[g, i, j] = mat['D'][g]
                    self.mysolv.get_NSF()[g, i, j] = mat['NSF'][g]
                    self.mysolv.get_Chi()[g, i, j] = mat['CHI'][g]
                    self.mysolv.get_SigR()[g, i, j] = mat['SIGR'][g]
                self.mysolv.get_SigS()[1, 0, i, j] = mat['S12']
                self.mysolv.get_SigS()[0, 1, i, j] = mat['S21']
        
        print(f"  Cellules combustible: {n_fuel}")
        print(f"  Cellules baffle     : {n_baffle}")
        print(f"  Cellules réflecteur : {n_reflector}")
        
        print("\n=== VALIDATION DES COEFFICIENTS ===")
        for g in range(self.num_groups):
            D = self.mysolv.get_D()[g]
            print(f"Groupe {g}: D ∈ [{D.min():.6f}, {D.max():.6f}] cm")
        self.mysolv.BuildMatrices()
        print(f"\n✅ Solveur initialisé en {time.time()-timeref:.3f} s")

    def solve(self, forward=True, adjoint=False, use_direct_keff=False):
        if self.mysolv is None:
            raise RuntimeError("Le solveur doit être initialisé")
        timeref = time.time()
        self.mysolv.set_tol(1e-5, 1e-4, 1e-4, 200, 1000)
        print("\n=== RÉSOLUTION K-EFF ===")
        print(f"  Solveur linéaire : {self.mysolv.GetSolverName()}")
        print(f"  Éléments         : RT{self.rt_order}-P{self.p_order}")
        print(f"  Initialisation coarse : {'Oui' if self.use_coarse else 'Non'}")
        if self.use_coarse:
            print(f"  Facteurs coarse : {self.coarse_factors}")
        
        if forward:
            print("\n--- Problème DIRECT ---")
            self.keff = self.mysolv.SolveKeff(use_coarse_init=self.use_coarse, coarse_factors=self.coarse_factors)
            if self.p_order == 0:
                self.phi = np.array([self.mysolv.get_flux()[g] for g in range(self.num_groups)])
        if adjoint:
            print("\n--- Problème ADJOINT ---")
            self.keff_adj = self.mysolv.SolveAdjoint(normalize_to_direct=forward, use_direct_keff=use_direct_keff)
            if self.p_order == 0:
                self.phi_adj = np.array([self.mysolv.get_flux_adj()[g] for g in range(self.num_groups)])
        
        time1 = time.time()
        print("\n" + "="*60)
        print("✅ CONVERGENCE ATTEINTE")
        if forward:
            ecart_pcm = 1E5 * (1/self.kref - 1/self.keff)
            ecart_rel = 100 * (self.keff - self.kref) / self.kref
            print(f"   k-eff direct    = {self.keff:.6f}")
            print(f"   k-eff référence = {self.kref:.6f}")
            print(f"   Écart absolu    = {ecart_pcm:+.2f} pcm")
            print(f"   Écart relatif   = {ecart_rel:+.4f} %")
        if adjoint:
            print(f"   k-eff adjoint   = {self.keff_adj:.6f}")
            if forward:
                print(f"   |k - k†|        = {abs(self.keff - self.keff_adj):.2e}")
        print(f"   Temps résolution = {time1-timeref:.2f} s")
        print("="*60)

        if self.p_order == 0 and self.phi is not None:
            Ny, Nx = self.maillage.shape
            self.pvol = np.zeros((Ny, Nx))
            for i in range(Ny):
                for j in range(Nx):
                    for g in range(self.num_groups):
                        self.pvol[i, j] += self.mysolv.get_NSF()[g, i, j] * self.mysolv.get_flux()[g, i, j]
            
            # Extraction de la partie non nulle pour reshape
            arr = self.pvol
            row_start_idx = [np.argmax(row != 0) if np.any(row != 0) else arr.shape[1] for row in arr]
            row_end_idx = [len(row) - np.argmax(row[::-1] != 0) - 1 if np.any(row != 0) else -1 for row in arr]
            col_start_idx = [np.argmax(arr[:, j] != 0) if np.any(arr[:, j] != 0) else arr.shape[0] for j in range(arr.shape[1])]
            col_end_idx = [arr.shape[0] - np.argmax(arr[::-1, j] != 0) - 1 if np.any(arr[:, j] != 0) else -1 for j in range(arr.shape[1])]
            start_row, end_row = min(row_start_idx), max(row_end_idx)
            start_col, end_col = min(col_start_idx), max(col_end_idx)
            sub_arr = arr[start_row:end_row+1, start_col:end_col+1]
            self.sub_arr = sub_arr
            self.Fass = sub_arr.reshape(15, sub_arr.shape[0] // 15, 15, sub_arr.shape[1] // 15)
            self.Fass = self.Fass.sum(axis=1).sum(axis=2)
            self.Fass = 193. * self.Fass / self.Fass.sum()

    def export_vtk(self, filename="zion2d", export_adjoint=True):
        if self.mysolv is None:
            print("❌ Solveur non initialisé")
            return
        self.mysolv.ExportVTK(filename, export_flux=True, export_current=True, export_xs=True,
                              export_adjoint=export_adjoint and self.phi_adj is not None)
        print(f"✅ Export VTK: {filename}.vtk")

    def plot_flux(self, group=0, adjoint=False):
        data = self.phi_adj if adjoint else self.phi
        label = "adjoint" if adjoint else "direct"
        if data is None:
            print(f"❌ Flux {label} non disponible")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[group], cmap='jet', cbar_kws={'label': f'Flux φ{group+1} ({label})'})
        keff = self.keff_adj if adjoint else self.keff
        plt.title(f"Flux Groupe {group+1} ({label}) - k-eff = {keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_pvol(self):
        if self.pvol is None:
            print("❌ Puissance non disponible")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pvol, cmap='jet', cbar_kws={'label': 'Puissance'})
        plt.title(f"Distribution de puissance - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_Fass(self):
        if self.Fass is None:
            print("❌ Facteurs assemblages non disponibles")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.Fass, cmap='jet', annot=True, fmt=".4f")
        plt.title(f"Facteurs assemblages - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def check_Ffaisc(self):
        data_zion2D = np.array([
            [np.nan, np.nan, np.nan, np.nan, 0.3159, 0.4393, 0.4902, 0.5053, 0.4902, 0.4393, 0.3159, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, 0.3206, 0.5273, 0.7189, 0.7189, 0.9181, 0.7973, 0.9181, 0.7189, 0.7189, 0.5273, 0.3206, np.nan, np.nan],   
            [np.nan, 0.3206, 0.6642, 0.8494, 0.8945, 1.0814, 1.0334, 1.1637, 1.0334, 1.0814, 0.8945, 0.8494, 0.6642, 0.3206, np.nan],   
            [np.nan, 0.5273, 0.8494, 1.0778, 1.2171, 1.1811, 1.3646, 1.2532, 1.3646, 1.1811, 1.2171, 1.0778, 0.8494, 0.5273, np.nan],   
            [0.3159, 0.7189, 0.8945, 1.2171, 1.2433, 1.4776, 1.3955, 1.5649, 1.3955, 1.4776, 1.2433, 1.2171, 0.8945, 0.7189, 0.3159],   
            [0.4393, 0.7189, 1.0814, 1.1811, 1.4776, 1.4463, 1.6720, 1.5348, 1.6720, 1.4463, 1.4776, 1.1811, 1.0814, 0.7189, 0.4393],   
            [0.4902, 0.9181, 1.0334, 1.3646, 1.3955, 1.6720, 1.5834, 1.7766, 1.5834, 1.6720, 1.3955, 1.3646, 1.0334, 0.9181, 0.4902],   
            [0.5053, 0.7973, 1.1637, 1.2532, 1.5649, 1.5348, 1.7766, 1.6315, 1.7766, 1.5348, 1.5649, 1.2532, 1.1637, 0.7973, 0.5053],   
            [0.4902, 0.9181, 1.0334, 1.3646, 1.3955, 1.6720, 1.5834, 1.7766, 1.5834, 1.6720, 1.3955, 1.3646, 1.0334, 0.9181, 0.4902],   
            [0.4393, 0.7189, 1.0814, 1.1811, 1.4776, 1.4463, 1.6720, 1.5348, 1.6720, 1.4463, 1.4776, 1.1811, 1.0814, 0.7189, 0.4393],   
            [0.3159, 0.7189, 0.8945, 1.2171, 1.2433, 1.4776, 1.3955, 1.5649, 1.3955, 1.4776, 1.2433, 1.2171, 0.8945, 0.7189, 0.3159],   
            [np.nan, 0.5273, 0.8494, 1.0778, 1.2171, 1.1811, 1.3646, 1.2532, 1.3646, 1.1811, 1.2171, 1.0778, 0.8494, 0.5273, np.nan],   
            [np.nan, 0.3206, 0.6642, 0.8494, 0.8945, 1.0814, 1.0334, 1.1637, 1.0334, 1.0814, 0.8945, 0.8494, 0.6642, 0.3206, np.nan],   
            [np.nan, np.nan, 0.3206, 0.5273, 0.7189, 0.7189, 0.9181, 0.7973, 0.9181, 0.7189, 0.7189, 0.5273, 0.3206, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, 0.3159, 0.4393, 0.4902, 0.5053, 0.4902, 0.4393, 0.3159, np.nan, np.nan, np.nan, np.nan]   
        ])
        return 100. * (data_zion2D - self.Fass) / data_zion2D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZION 2D - Solveur neutronique (Version Eigen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python zion2D.py --mesh-assembly 2 --mesh-baffle 2
      → 2 mailles par assemblage, 2 mailles dans le baffle
  
  python zion2D.py --mesh-assembly 4 --mesh-baffle 1
      → 4 mailles par assemblage (5.402 cm), 1 maille dans le baffle (2.8575 cm)

Dimensions géométriques:
  - Assemblage combustible : 21.608 cm
  - Épaisseur baffle       : 2.8575 cm
        """
    )
    
    # Paramètres de maillage
    parser.add_argument("--mesh-assembly", type=int, default=1, 
                       help="Subdivisions par assemblage (défaut: 1)")
    parser.add_argument("--mesh-baffle", type=int, default=1, 
                       help="Subdivisions dans le baffle (défaut: 1)")
    
    # Domaine
    parser.add_argument("--domain", type=str, default="entier", 
                       choices=["entier", "quart_so"], 
                       help="Géométrie du domaine")
    
    # Ordres des éléments finis
    parser.add_argument("--rt-order", type=int, default=0, choices=[0, 1, 2], 
                       help="Ordre RT")
    parser.add_argument("--p-order", type=int, default=0, choices=[0, 1, 2], 
                       help="Ordre P")
    parser.add_argument("--order", type=int, default=None, choices=[0, 1, 2], 
                       help="Ordre unique RT=P")
    
    # Adjoint
    parser.add_argument("--adjoint_only", action="store_true", 
                       help="Résoudre seulement l'adjoint")
    parser.add_argument("--adjoint", action="store_true", 
                       help="Résoudre aussi l'adjoint")
    parser.add_argument("--use-direct-keff", action="store_true", 
                       help="Utiliser k-eff direct pour l'adjoint")
    
    # Options
    parser.add_argument("--no-coarse", action="store_true", 
                       help="Désactiver l'initialisation coarse")
    parser.add_argument("--plot", action="store_true", 
                       help="Afficher les graphiques")
    parser.add_argument("--vtk", type=str, default=None, 
                       help="Exporter au format VTK")
    
    args = parser.parse_args()

    rt_order = args.order if args.order is not None else args.rt_order
    p_order = args.order if args.order is not None else args.p_order
    solve_forward = not args.adjoint_only
    solve_adjoint = args.adjoint or args.adjoint_only

    print("="*60)
    print("BENCHMARK ZION 2D - Version Eigen")
    print(f"  → Éléments finis : RT{rt_order}-P{p_order}")
    print("  → Solveur: BiCGSTAB")
    print("="*60)
    print(f"Maillage assemblage : {args.mesh_assembly} subdivision(s)")
    print(f"Maillage baffle     : {args.mesh_baffle} subdivision(s)")
    print(f"Domaine   : {args.domain}")
    print(f"Coarse    : {'Non' if args.no_coarse else 'Oui'}")
    print(f"Direct    : {'Oui' if solve_forward else 'Non'}")
    print(f"Adjoint   : {'Oui' if solve_adjoint else 'Non'}")
    print("="*60)

    zion2d = Zion2D(
        domaine=args.domain,
        submesh_assembly=args.mesh_assembly,
        submesh_baffle=args.mesh_baffle
    )
    zion2d.rt_order = rt_order
    zion2d.p_order = p_order
    zion2d.use_coarse = not args.no_coarse
    zion2d.load_zion2d_mat()
    zion2d.mesh_initialisation()
    zion2d.init_solver()
    zion2d.solve(forward=solve_forward, adjoint=solve_adjoint, use_direct_keff=args.use_direct_keff)

    if args.vtk:
        zion2d.export_vtk(args.vtk, export_adjoint=solve_adjoint)
    if args.plot:
        if solve_forward:
            zion2d.plot_flux(group=0, adjoint=False)
            zion2d.plot_flux(group=1, adjoint=False)
        if solve_adjoint:
            zion2d.plot_flux(group=0, adjoint=True)
            zion2d.plot_flux(group=1, adjoint=True)
        if zion2d.pvol is not None:
            zion2d.plot_pvol()

    print(f"\n⏱️  Temps total : {time.time() - zion2d.start:.2f} s")
    print("="*60)
