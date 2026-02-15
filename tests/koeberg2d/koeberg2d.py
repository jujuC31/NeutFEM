"""
BENCHMARK KOEBERG 2D - Version Eigen (sans MFEM)
Adaptée pour neutfem_eigen avec:
  - Ordres mixtes RT_k-P_m
  - Solveur adjoint
  - Export VTK
  - Initialisation coarse mesh
  - Solveur BiCGSTAB (sans préconditionneur)
  - 4 groupes d'énergie
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


class Koeberg2D:
    """
    Benchmark KOEBERG 2D avec fonctionnalités avancées
    Version adaptée pour NeutFEM Eigen - 4 groupes d'énergie
    """
    
    def __init__(self, meshtype="2x2", domaine="entier", ncpu=1, sym="cyclique"):
        
        self.start = time.time()
        self.meshtype = meshtype
        self.domaine = domaine
        self.ncpu = ncpu
        self.sym = sym
        
        self.kref = 1.007954  # k-eff référence Koeberg
        self.num_groups = 4   # 4 groupes d'énergie
        self.verbose = 0
        
        # Ordres des éléments finis
        self.rt_order = 0
        self.p_order = 0
        
        self.init_meshing = False
        self.mysolv = None
        self.keff = None
        self.keff_adj = None
        self.phi = None
        self.phi_adj = None
        self.pvol = None
        
        # Paramètres coarse mesh
        self.use_coarse = True
        self.coarse_factors = [2, 2, 1]

        # Maillage 17×17 assemblages
        self.maillage_motifs_coeur = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "F7", "F7", "F7", "F7", "F7", "  ", "  ", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "F7", "F7", "F7", "F4", "F4", "F4", "F7", "F7", "F7", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "F7", "F7", "F4", "F4", "F6", "F1", "F6", "F4", "F4", "F7", "F7", "  ", "  ", "  "],  
            ["  ", "  ", "F7", "F7", "F4", "F5", "F3", "F1", "F2", "F1", "F3", "F5", "F4", "F7", "F7", "  ", "  "],  
            ["  ", "F7", "F7", "F4", "F1", "F3", "F1", "F2", "F1", "F2", "F1", "F3", "F1", "F4", "F7", "F7", "  "],  
            ["  ", "F7", "F4", "F5", "F3", "F1", "F2", "F1", "F3", "F1", "F2", "F1", "F3", "F5", "F4", "F7", "  "],  
            ["F7", "F7", "F4", "F3", "F1", "F2", "F1", "F3", "F1", "F3", "F1", "F2", "F1", "F3", "F4", "F7", "F7"],
            ["F7", "F4", "F6", "F1", "F2", "F1", "F3", "F1", "F3", "F1", "F3", "F1", "F2", "F1", "F6", "F4", "F7"],
            ["F7", "F4", "F1", "F2", "F1", "F3", "F1", "F3", "F1", "F3", "F1", "F3", "F1", "F2", "F1", "F4", "F7"],
            ["F7", "F4", "F6", "F1", "F2", "F1", "F3", "F1", "F3", "F1", "F3", "F1", "F2", "F1", "F6", "F4", "F7"],
            ["F7", "F7", "F4", "F3", "F1", "F2", "F1", "F3", "F1", "F3", "F1", "F2", "F1", "F3", "F4", "F7", "F7"],  
            ["  ", "F7", "F4", "F5", "F3", "F1", "F2", "F1", "F3", "F1", "F2", "F1", "F3", "F5", "F4", "F7", "  "],  
            ["  ", "F7", "F7", "F4", "F1", "F3", "F1", "F2", "F1", "F2", "F1", "F3", "F1", "F4", "F7", "F7", "  "],  
            ["  ", "  ", "F7", "F7", "F4", "F5", "F3", "F1", "F2", "F1", "F3", "F5", "F4", "F7", "F7", "  ", "  "],  
            ["  ", "  ", "  ", "F7", "F7", "F4", "F4", "F6", "F1", "F6", "F4", "F4", "F7", "F7", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "F7", "F7", "F7", "F4", "F4", "F4", "F7", "F7", "F7", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F7", "F7", "F7", "F7", "F7", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

    def plot_geom(self):
        """Visualise la géométrie"""
        if not self.init_meshing:
            print("❌ Erreur: Le maillage doit être initialisé d'abord")
            return

        maillage_draw = []
        for row in self.maillage:
            maillage_draw.append([
                0 if cell == "  " else int(cell[1]) 
                for cell in row
            ])

        sns.heatmap(maillage_draw, cmap='jet', linewidths=0.5, linecolor="k")
        plt.title(f"Géométrie Koeberg - {self.meshtype} - {self.domaine}")
        plt.show()

    def plot_materials(self):
        """Visualise les matériaux du cœur"""
        maillage_draw = []
        for row in self.maillage_motifs_coeur:
            maillage_draw.append([
                0 if cell == "  " else int(cell[1]) 
                for cell in row
            ])

        sns.heatmap(maillage_draw, cmap='jet', annot=True, 
                   linewidths=1, linecolor="k", fmt='d')
        plt.title("Distribution des matériaux Koeberg 2D")
        plt.show()

    def mesh_initialisation(self, meshtype=None, domaine=None):
        """Initialise le maillage"""
        timeref = time.time()

        if meshtype:
            self.meshtype = meshtype
        if domaine:
            self.domaine = domaine
        
        if "x" in self.meshtype:
            self.nmeshes = int(self.meshtype.split("x")[0])
        else:
            self.nmeshes = len(self.meshtype)

        # Expansion
        self.maillage = np.array([
            [cell for cell in row for _ in range(self.nmeshes)] 
            for row in self.maillage_motifs_coeur 
            for _ in range(self.nmeshes)
        ])

        # Symétries
        L = len(self.maillage)
        L_half = L // 2
        
        domaine_map = {
            "quart_so": (slice(L_half, None), slice(None, L_half)),
            "quart_no": (slice(None, L_half), slice(None, L_half)),
            "quart_ne": (slice(None, L_half), slice(L_half, None)),
            "quart_se": (slice(L_half, None), slice(L_half, None)),
            "moitie_s": (slice(L_half, None), slice(None, None)),
            "moitie_o": (slice(None, None), slice(None, L_half)),
            "moitie_n": (slice(None, L_half), slice(None, None)),
            "moitie_e": (slice(None, None), slice(L_half, None)),
        }
        
        if self.domaine in domaine_map:
            y_slice, x_slice = domaine_map[self.domaine]
            self.maillage = self.maillage[y_slice, x_slice]

        # Coordonnées (taille assemblage = 21.608 cm)
        cell_size = 21.608 / self.nmeshes
        nx_cells = self.maillage.shape[1]
        ny_cells = self.maillage.shape[0]

        self.x_breaks = np.linspace(0.0, nx_cells * cell_size, nx_cells + 1)
        self.y_breaks = np.linspace(0.0, ny_cells * cell_size, ny_cells + 1)
        self.z_breaks = np.array([0.0])
        
        # Calcul automatique des facteurs coarse
        self._compute_coarse_factors(nx_cells, ny_cells)
        
        time1 = time.time()
        print(f"✅ Maillage initialisé: {ny_cells}×{nx_cells} cellules "
              f"({time1-timeref:.3f} s)")
        self.init_meshing = True

    def _compute_coarse_factors(self, nx, ny):
        """Calcule les facteurs de réduction coarse optimaux"""
        def find_factor(n, max_factor=4):
            for f in range(min(max_factor, n), 0, -1):
                if n % f == 0:
                    return f
            return 1
        
        rx = find_factor(nx)
        ry = find_factor(ny)
        self.coarse_factors = [rx, ry, 1]
        print(f"  Facteurs coarse calculés: {rx}×{ry}×1")

    def load_koeberg2d_mat(self):
        """Charge les matériaux Koeberg 2D (4 groupes)"""
        
        ng = self.num_groups
        
        # F1
        self.F1 = {
            'D': [2.491869, 1.045224, 0.677407, 0.375191],
            'ABS': [0.003654, 0.002124, 0.019908, 0.067990],
            'NSF': [0.008228, 0.000536, 0.007058, 0.083930],
            'CHI': [0.745248, 0.254328, 0.000424, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F1['SCATTER'][1, 0] = 0.063789
        self.F1['SCATTER'][2, 0] = 0.000486
        self.F1['SCATTER'][2, 1] = 0.064381
        self.F1['SCATTER'][3, 1] = 0.000003
        self.F1['SCATTER'][3, 2] = 0.050849
        self.F1['SCATTER'][2, 3] = 0.001245
        self.F1['SIGR'] = [self.F1['ABS'][g] + sum(self.F1['SCATTER'][:, g]) - self.F1['SCATTER'][g, g] 
                          for g in range(ng)]

        # F2
        self.F2 = {
            'D': [2.492653, 1.049844, 0.676610, 0.379481],
            'ABS': [0.003685, 0.002215, 0.022012, 0.085052],
            'NSF': [0.008295, 0.000713, 0.009230, 0.108244],
            'CHI': [0.745248, 0.254328, 0.000424, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F2['SCATTER'][1, 0] = 0.063112
        self.F2['SCATTER'][2, 0] = 0.000478
        self.F2['SCATTER'][2, 1] = 0.063078
        self.F2['SCATTER'][3, 1] = 0.000003
        self.F2['SCATTER'][3, 2] = 0.048420
        self.F2['SCATTER'][2, 3] = 0.001543
        self.F2['SIGR'] = [self.F2['ABS'][g] + sum(self.F2['SCATTER'][:, g]) - self.F2['SCATTER'][g, g] 
                          for g in range(ng)]

        # F3
        self.F3 = {
            'D': [2.491978, 1.051910, 0.677084, 0.381453],
            'ABS': [0.003684, 0.002221, 0.022403, 0.088077],
            'NSF': [0.008285, 0.000713, 0.009214, 0.108087],
            'CHI': [0.745248, 0.254328, 0.000424, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F3['SCATTER'][1, 0] = 0.062765
        self.F3['SCATTER'][2, 0] = 0.000473
        self.F3['SCATTER'][2, 1] = 0.062404
        self.F3['SCATTER'][3, 1] = 0.000003
        self.F3['SCATTER'][3, 2] = 0.047549
        self.F3['SCATTER'][2, 3] = 0.001598
        self.F3['SIGR'] = [self.F3['ABS'][g] + sum(self.F3['SCATTER'][:, g]) - self.F3['SCATTER'][g, g] 
                          for g in range(ng)]

        # F4
        self.F4 = {
            'D': [2.492535, 1.045298, 0.674684, 0.374240],
            'ABS': [0.003740, 0.002299, 0.022621, 0.091000],
            'NSF': [0.008459, 0.000923, 0.011714, 0.133600],
            'CHI': [0.745248, 0.254328, 0.000424, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F4['SCATTER'][1, 0] = 0.062737
        self.F4['SCATTER'][2, 0] = 0.000486
        self.F4['SCATTER'][2, 1] = 0.064330
        self.F4['SCATTER'][3, 1] = 0.000003
        self.F4['SCATTER'][3, 2] = 0.049518
        self.F4['SCATTER'][2, 3] = 0.001630
        self.F4['SIGR'] = [self.F4['ABS'][g] + sum(self.F4['SCATTER'][:, g]) - self.F4['SCATTER'][g, g] 
                          for g in range(ng)]

        # F5
        self.F5 = {
            'D': [2.492329, 1.051953, 0.675683, 0.380606],
            'ABS': [0.003730, 0.002315, 0.023822, 0.100246],
            'NSF': [0.008409, 0.000921, 0.011675, 0.134282],
            'CHI': [0.745248, 0.254328, 0.000424, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F5['SCATTER'][1, 0] = 0.062737
        self.F5['SCATTER'][2, 0] = 0.000473
        self.F5['SCATTER'][2, 1] = 0.062376
        self.F5['SCATTER'][3, 1] = 0.000003
        self.F5['SCATTER'][3, 2] = 0.046859
        self.F5['SCATTER'][2, 3] = 0.001797
        self.F5['SIGR'] = [self.F5['ABS'][g] + sum(self.F5['SCATTER'][:, g]) - self.F5['SCATTER'][g, g] 
                          for g in range(ng)]

        # F6
        self.F6 = {
            'D': [2.491521, 1.054029, 0.676197, 0.382434],
            'ABS': [0.003730, 0.002321, 0.024196, 0.103283],
            'NSF': [0.008400, 0.000921, 0.011651, 0.133974],
            'CHI': [0.745248, 0.254328, 0.000424, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F6['SCATTER'][1, 0] = 0.062386
        self.F6['SCATTER'][2, 0] = 0.000468
        self.F6['SCATTER'][2, 1] = 0.061696
        self.F6['SCATTER'][3, 1] = 0.000003
        self.F6['SCATTER'][3, 2] = 0.046005
        self.F6['SCATTER'][2, 3] = 0.001852
        self.F6['SIGR'] = [self.F6['ABS'][g] + sum(self.F6['SCATTER'][:, g]) - self.F6['SCATTER'][g, g] 
                          for g in range(ng)]

        # F7 - Réflecteur
        self.F7 = {
            'D': [2.119737, 0.980098, 0.531336, 1.058029],
            'ABS': [0.000466, 0.000263, 0.004282, 0.116918],
            'NSF': [0.0, 0.0, 0.0, 0.0],
            'CHI': [0.0, 0.0, 0.0, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.F7['SCATTER'][1, 0] = 0.042052
        self.F7['SCATTER'][2, 0] = 0.000322
        self.F7['SCATTER'][2, 1] = 0.044589
        self.F7['SCATTER'][3, 2] = 0.052246
        self.F7['SCATTER'][2, 3] = 0.000978
        self.F7['SIGR'] = [self.F7['ABS'][g] + sum(self.F7['SCATTER'][:, g]) - self.F7['SCATTER'][g, g] 
                          for g in range(ng)]

        # R0 - Vide (absorbant fort)
        self.R0 = {
            'D': [0.1 * d for d in self.F7['D']],
            'ABS': [1e8, 1e8, 1e8, 1e8],
            'NSF': [0.0, 0.0, 0.0, 0.0],
            'CHI': [0.0, 0.0, 0.0, 0.0],
            'SCATTER': np.zeros((ng, ng))
        }
        self.R0['SIGR'] = self.R0['ABS'].copy()
        
        print("✅ Matériaux chargés: F1-F7 + R0 (4 groupes)")

    def init_solver(self):
        """Initialise le solveur NeutFEM"""
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé d'abord")
        
        timeref = time.time()
        
        order_str = f"RT{self.rt_order}-P{self.p_order}"
        print(f"\n=== INITIALISATION SOLVEUR {order_str} ===")
        
        # Création du solveur
        if self.rt_order == self.p_order:
            self.mysolv = neutron_solver.NeutFEM(
                self.rt_order,
                self.num_groups,
                self.x_breaks,
                self.y_breaks,
                self.z_breaks
            )
        else:
            self.mysolv = neutron_solver.NeutFEM(
                self.rt_order,
                self.p_order,
                self.num_groups,
                self.x_breaks,
                self.y_breaks,
                self.z_breaks
            )
        
        # Configuration solveur linéaire
        self.mysolv.set_linear_solver(LinearSolverType.BICGSTAB)
        
        # Conditions aux limites selon domaine
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
        
        elif self.domaine == "moitie_s":
            self.mysolv.apply_central_symmetry(0, 1)
            self.mysolv.set_bc(int(BoundaryID.TOP_2D), BCType.MIRROR, 0.0)
            self.mysolv.set_bc(int(BoundaryID.LEFT_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.RIGHT_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.BOTTOM_2D), BCType.DIRICHLET, 0.0)
            print("  Domaine MOITIE_S : Symétrie centrale activée")
        
        elif self.domaine == "moitie_o":
            self.mysolv.apply_central_symmetry(1, 0)
            self.mysolv.set_bc(int(BoundaryID.RIGHT_2D), BCType.MIRROR, 0.0)
            self.mysolv.set_bc(int(BoundaryID.LEFT_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.TOP_2D), BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(int(BoundaryID.BOTTOM_2D), BCType.DIRICHLET, 0.0)
            print("  Domaine MOITIE_O : Symétrie centrale activée")

        # Remplissage des coefficients
        Ny, Nx = self.maillage.shape
        print(f"\n=== REMPLISSAGE DES COEFFICIENTS ({Ny}×{Nx}) ===")
        
        for i in range(Ny):
            for j in range(Nx):
                fuel_key = self.maillage[i, j].strip()
                mat = getattr(self, fuel_key) if hasattr(self, fuel_key) and fuel_key else self.R0
                
                for g in range(self.num_groups):
                    self.mysolv.get_D()[g, i, j] = mat['D'][g]
                    self.mysolv.get_NSF()[g, i, j] = mat['NSF'][g]
                    self.mysolv.get_Chi()[g, i, j] = mat['CHI'][g]
                    self.mysolv.get_SigR()[g, i, j] = mat['SIGR'][g]
                
                # Remplissage de la matrice de scattering
                for g_to in range(self.num_groups):
                    for g_from in range(self.num_groups):
                        self.mysolv.get_SigS()[g_to, g_from, i, j] = mat['SCATTER'][g_to, g_from]
        
        # Validation
        print("\n=== VALIDATION DES COEFFICIENTS ===")
        for g in range(self.num_groups):
            D = self.mysolv.get_D()[g]
            print(f"Groupe {g}: D ∈ [{D.min():.6f}, {D.max():.6f}] cm")
        
        # Construction des matrices
        self.mysolv.BuildMatrices()

        time1 = time.time()
        print(f"\n✅ Solveur initialisé en {time1-timeref:.3f} s")

    def solve(self, forward=True, adjoint=False, use_direct_keff=False):
        """Résout le problème avec initialisation coarse mesh"""
        if self.mysolv is None:
            raise RuntimeError("Le solveur doit être initialisé")
        
        timeref = time.time()
        
        self.mysolv.set_tol(1e-5, 1e-4, 1e-4, 200, 1000)

        print("\n=== RÉSOLUTION K-EFF ===")
        print(f"  Solveur linéaire : {self.mysolv.GetSolverName()}")
        print(f"  Éléments         : RT{self.rt_order}-P{self.p_order}")
        print(f"  Groupes          : {self.num_groups}")
        print(f"  Initialisation coarse : {'Oui' if self.use_coarse else 'Non'}")
        if self.use_coarse:
            print(f"  Facteurs coarse : {self.coarse_factors}")
        
        # Résolution directe
        if forward:
            print("\n--- Problème DIRECT ---")
            self.keff = self.mysolv.SolveKeff(
                use_coarse_init=self.use_coarse,
                coarse_factors=self.coarse_factors
            )
            
            if self.p_order == 0:
                self.phi = np.array([
                    self.mysolv.get_flux()[g] for g in range(self.num_groups)
                ])

        # Résolution adjointe
        if adjoint:
            print("\n--- Problème ADJOINT ---")
            self.keff_adj = self.mysolv.SolveAdjoint(
                normalize_to_direct=forward,
                use_direct_keff=use_direct_keff
            )
            
            if self.p_order == 0:
                self.phi_adj = np.array([
                    self.mysolv.get_flux_adj()[g] for g in range(self.num_groups)
                ])
        
        time1 = time.time()
        
        # Affichage résultats
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
                diff_adj = abs(self.keff - self.keff_adj)
                print(f"   |k - k†|        = {diff_adj:.2e}")
        
        print(f"   Temps résolution = {time1-timeref:.2f} s")
        print("="*60)

        # Puissance (seulement pour P0)
        if self.p_order == 0 and self.phi is not None:
            Ny, Nx = self.maillage.shape
            self.pvol = np.zeros((Ny, Nx))
            
            for i in range(Ny):
                for j in range(Nx):
                    for g in range(self.num_groups):
                        nsf_g = self.mysolv.get_NSF()[g]
                        phi_g = self.mysolv.get_flux()[g]
                        self.pvol[i, j] += nsf_g[i, j] * phi_g[i, j]
            
            # Facteurs assemblages
            self.Fass = self.pvol.reshape((17, self.nmeshes, 17, self.nmeshes))
            self.Fass = self.Fass.sum(axis=1).sum(axis=2)
            self.Fass = 157. * self.Fass / self.Fass.sum()

    def export_vtk(self, filename="koeberg2d", export_adjoint=True):
        """Exporte les résultats au format VTK"""
        if self.mysolv is None:
            print("❌ Solveur non initialisé")
            return
        
        self.mysolv.ExportVTK(
            filename,
            export_flux=True,
            export_current=True,
            export_xs=True,
            export_adjoint=export_adjoint and self.phi_adj is not None
        )
        print(f"✅ Export VTK: {filename}.vtk")

    def plot_flux(self, group=0, adjoint=False):
        """Visualise le flux"""
        data = self.phi_adj if adjoint else self.phi
        label = "adjoint" if adjoint else "direct"
        
        if data is None:
            print(f"❌ Flux {label} non disponible")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[group], cmap='jet', 
                   cbar_kws={'label': f'Flux φ{group+1} ({label})'})
        keff = self.keff_adj if adjoint else self.keff
        plt.title(f"Flux Groupe {group+1} ({label}) - k-eff = {keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_pvol(self):
        """Visualise la puissance"""
        if self.pvol is None:
            print("❌ Puissance non disponible")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pvol, cmap='jet', 
                   cbar_kws={'label': 'Puissance'})
        plt.title(f"Distribution de puissance - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_Fass(self):
        """Visualise les facteurs assemblages"""
        if self.Fass is None:
            print("❌ Facteurs assemblages non disponibles")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.Fass, cmap='jet', annot=True, fmt=".4f")
        plt.title(f"Facteurs assemblages - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def check_Ffaisc(self):
        """Vérifie les facteurs de forme par rapport à la référence"""
        data_koeberg2D = np.array([
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6425, 0.8331, 0.6425, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.6504, 0.9684, 1.0420, 0.9596, 1.0420, 0.9684, 0.6504, np.nan, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, 0.6670, 0.9813, 1.0390, 1.0617, 1.2147, 1.0617, 1.0390, 0.9813, 0.6670, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, 0.6670, 0.7860, 0.9988, 1.0581, 1.2430, 1.1319, 1.2430, 1.0581, 0.9988, 0.7860, 0.6670, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, 0.6504, 0.9813, 0.9988, 1.0363, 1.2236, 1.1054, 1.1639, 1.1054, 1.2236, 1.0363, 0.9988, 0.9813, 0.6504, np.nan, np.nan],   
            [np.nan, np.nan, 0.9684, 1.0390, 1.0581, 1.2236, 1.0929, 1.1305, 1.0445, 1.1305, 1.0929, 1.2236, 1.0581, 1.0390, 0.9684, np.nan, np.nan],   
            [np.nan, 0.6425, 1.0420, 1.0617, 1.2430, 1.1054, 1.1305, 1.0263, 1.0858, 1.0263, 1.1305, 1.1054, 1.2430, 1.0617, 1.0420, 0.6425, np.nan],   
            [np.nan, 0.8331, 0.9596, 1.2147, 1.1319, 1.1639, 1.0445, 1.0858, 1.0058, 1.0858, 1.0445, 1.1639, 1.1319, 1.2147, 0.9596, 0.8331, np.nan],   
            [np.nan, 0.6425, 1.0420, 1.0617, 1.2430, 1.1054, 1.1305, 1.0263, 1.0858, 1.0263, 1.1305, 1.1054, 1.2430, 1.0617, 1.0420, 0.6425, np.nan],   
            [np.nan, np.nan, 0.9684, 1.0390, 1.0581, 1.2236, 1.0929, 1.1305, 1.0445, 1.1305, 1.0929, 1.2236, 1.0581, 1.0390, 0.9684, np.nan, np.nan],   
            [np.nan, np.nan, 0.6504, 0.9813, 0.9988, 1.0363, 1.2236, 1.1054, 1.1639, 1.1054, 1.2236, 1.0363, 0.9988, 0.9813, 0.6504, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, 0.6670, 0.7860, 0.9988, 1.0581, 1.2430, 1.1319, 1.2430, 1.0581, 0.9988, 0.7860, 0.6670, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, 0.6670, 0.9813, 1.0390, 1.0617, 1.2147, 1.0617, 1.0390, 0.9813, 0.6670, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.6504, 0.9684, 1.0420, 0.9596, 1.0420, 0.9684, 0.6504, np.nan, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.6425, 0.8331, 0.6425, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],   
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],   
        ])
        
        diff = 100. * (data_koeberg2D - self.Fass) / data_koeberg2D
        return diff


# ===== SCRIPT PRINCIPAL =====
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="KOEBERG 2D - Solveur neutronique 4 groupes (Version Eigen + Ordres mixtes)"
    )
    
    # Maillage et domaine
    parser.add_argument("--mesh", type=str, default="2x2", 
                       help="Résolution du maillage (ex: 2x2, 4x4, 9x9)")
    parser.add_argument("--domain", type=str, default="entier",
                       choices=["entier", "quart_so", "moitie_s", "moitie_o"],
                       help="Géométrie du domaine")
    
    # Ordres des éléments finis
    parser.add_argument("--rt-order", type=int, default=0, choices=[0, 1, 2],
                       help="Ordre RT (Raviart-Thomas) pour le courant (défaut: 0)")
    parser.add_argument("--p-order", type=int, default=0, choices=[0, 1, 2],
                       help="Ordre P pour le flux (défaut: 0)")
    parser.add_argument("--order", type=int, default=None, choices=[0, 1, 2],
                       help="Ordre unique RT=P (raccourci pour --rt-order et --p-order)")
    
    # Problème adjoint
    parser.add_argument("--adjoint_only", action="store_true",
                       help="Résoudre seulement le problème adjoint")
    parser.add_argument("--adjoint", action="store_true",
                       help="Résoudre aussi le problème adjoint")
    parser.add_argument("--use-direct-keff", action="store_true",
                       help="Utiliser k-eff direct pour l'adjoint")
    
    # Options
    parser.add_argument("--no-coarse", action="store_true",
                       help="Désactiver l'initialisation coarse")
    parser.add_argument("--coarse-factor", type=int, default=2,
                       help="Facteur de réduction coarse (défaut: 2)")
    
    # Sorties
    parser.add_argument("--plot", action="store_true",
                       help="Afficher les graphiques")
    parser.add_argument("--vtk", type=str, default=None,
                       help="Exporter au format VTK")
    
    args = parser.parse_args()
    
    # Gestion des ordres
    if args.order is not None:
        rt_order = args.order
        p_order = args.order
    else:
        rt_order = args.rt_order
        p_order = args.p_order
    
    # Gestion adjoint
    solve_forward = not args.adjoint_only
    solve_adjoint = args.adjoint or args.adjoint_only
    
    order_str = f"RT{rt_order}-P{p_order}"
    
    print("="*60)
    print("BENCHMARK KOEBERG 2D - Version Eigen (4 groupes)")
    print(f"  → Éléments finis : {order_str}")
    print("  → Solveur: BiCGSTAB (sans préconditionneur)")
    print("  → Initialisation: Coarse mesh")
    print("="*60)
    print(f"Maillage  : {args.mesh}")
    print(f"Domaine   : {args.domain}")
    print(f"Coarse    : {'Non' if args.no_coarse else f'Oui (facteur {args.coarse_factor})'}")
    print(f"Direct    : {'Oui' if solve_forward else 'Non'}")
    print(f"Adjoint   : {'Oui' if solve_adjoint else 'Non'}")
    if solve_adjoint:
        print(f"  → k-eff direct : {'Oui' if args.use_direct_keff else 'Non'}")
    print("="*60)
    
    # Création
    koeberg2d = Koeberg2D(meshtype=args.mesh, domaine=args.domain)
    koeberg2d.rt_order = rt_order
    koeberg2d.p_order = p_order
    koeberg2d.use_coarse = not args.no_coarse
    
    # Matériaux et maillage
    koeberg2d.load_koeberg2d_mat()
    koeberg2d.mesh_initialisation()
    
    # Override coarse factors si spécifié
    if args.coarse_factor != 2:
        nx = koeberg2d.maillage.shape[1]
        ny = koeberg2d.maillage.shape[0]
        if nx % args.coarse_factor == 0 and ny % args.coarse_factor == 0:
            koeberg2d.coarse_factors = [args.coarse_factor, args.coarse_factor, 1]
            print(f"  Facteurs coarse override: {koeberg2d.coarse_factors}")
        else:
            print(f"  ⚠️  Facteur {args.coarse_factor} ne divise pas {nx}×{ny}, utilisation auto")
    
    # Résolution
    koeberg2d.init_solver()
    koeberg2d.solve(
        forward=solve_forward,
        adjoint=solve_adjoint,
        use_direct_keff=args.use_direct_keff
    )
    
    # Export VTK
    if args.vtk:
        koeberg2d.export_vtk(args.vtk, export_adjoint=solve_adjoint)
    
    # Visualisation
    if args.plot:
        if solve_forward:
            for g in range(koeberg2d.num_groups):
                koeberg2d.plot_flux(group=g, adjoint=False)
        if solve_adjoint:
            for g in range(koeberg2d.num_groups):
                koeberg2d.plot_flux(group=g, adjoint=True)
        if koeberg2d.pvol is not None:
            koeberg2d.plot_pvol()

    print(f"\n⏱️  Temps total : {time.time() - koeberg2d.start:.2f} s")
    print("="*60)
