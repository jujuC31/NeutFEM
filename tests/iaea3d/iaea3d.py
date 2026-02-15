"""
BENCHMARK IAEA 3D - Version Eigen (sans MFEM)
Adaptée pour neutfem_eigen avec:
  - Ordres mixtes RT_k-P_m
  - Solveur adjoint
  - Export VTK
  - Initialisation coarse mesh
  - Solveur BiCGSTAB (sans préconditionneur)
  - Géométrie 3D avec plans axiaux
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


class Iaea3D:
    """
    Benchmark IAEA 3D avec fonctionnalités avancées
    Version adaptée pour NeutFEM Eigen - Cas 3D avec plans axiaux
    """
    
    def __init__(self, meshtype="2x2", nmeshes_z=1, domaine="entier", ncpu=1, sym="cyclique"):
        
        self.start = time.time()
        self.meshtype = meshtype
        self.nmeshes_z = nmeshes_z
        self.domaine = domaine
        self.ncpu = ncpu
        self.sym = sym
        
        self.kref = 1.029096  # k-eff référence IAEA 3D
        self.num_groups = 2
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

        # Plans axiaux (19 plans selon z)
        # FA - plan supérieur avec barres de contrôle partielles
        self.FA = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # FB - plan standard avec barres de contrôle
        self.FB = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # FC - plan sans barres de contrôle
        self.FC = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # FD - réflecteur (plan inférieur)
        self.FD = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # Stack 3D: FA, FB×4, FC×12, FD
        self.maillage_motifs_coeur = np.array([self.FA, self.FB, self.FB, self.FB, self.FB, 
                                                self.FC, self.FC, self.FC, self.FC, self.FC, 
                                                self.FC, self.FC, self.FC, self.FC, self.FC,
                                                self.FC, self.FC, self.FC, self.FD])

    def plot_geom(self, plan=0):
        if not self.init_meshing:
            print("❌ Erreur: Le maillage doit être initialisé d'abord")
            return
        maillage_draw = [[0 if c == "  " else int(c[1]) for c in row] for row in self.maillage[plan]]
        sns.heatmap(maillage_draw, cmap='jet', linewidths=0.5, linecolor="k")
        plt.title(f"Géométrie IAEA 3D - Plan {plan}")
        plt.show()

    def plot_materials(self, plan=0):
        maillage_draw = [[0 if c == "  " else int(c[1]) for c in row] for row in self.maillage_motifs_coeur[plan]]
        sns.heatmap(maillage_draw, cmap='jet', annot=True, linewidths=1, linecolor="k", fmt='d')
        plt.title(f"Distribution des matériaux IAEA 3D - Plan {plan}")
        plt.show()

    def mesh_initialisation(self, meshtype=None, domaine=None):
        timeref = time.time()
        if meshtype:
            self.meshtype = meshtype
        if domaine:
            self.domaine = domaine
        
        if "x" in self.meshtype:
            self.nmeshes = int(self.meshtype.split("x")[0])
        else:
            self.nmeshes = len(self.meshtype)

        # Expansion 3D
        self.maillage = np.array([
            [[c for c in row for _ in range(self.nmeshes)] for row in zcell for _ in range(self.nmeshes)]
            for zcell in self.maillage_motifs_coeur for _ in range(self.nmeshes_z)
        ])

        # Symétries (appliquées sur chaque plan)
        L_xy = self.maillage.shape[1]
        L_half = L_xy // 2
        
        if "quart" in self.domaine:
            domaine_map = {
                "quart_so": (slice(L_half, None), slice(None, L_half)),
                "quart_no": (slice(None, L_half), slice(None, L_half)),
                "quart_ne": (slice(None, L_half), slice(L_half, None)),
                "quart_se": (slice(L_half, None), slice(L_half, None)),
            }
            y_slice, x_slice = domaine_map.get(self.domaine, (slice(None), slice(None)))
            self.maillage = self.maillage[:, y_slice, x_slice]

        # Coordonnées
        cell_size_xy = 20.0 / self.nmeshes
        cell_size_z = 20.0 / self.nmeshes_z  # Hauteur totale 380 cm / 19 plans
        nz_cells = self.maillage.shape[0]
        ny_cells = self.maillage.shape[1]
        nx_cells = self.maillage.shape[2]

        self.x_breaks = np.linspace(0.0, nx_cells * cell_size_xy, nx_cells + 1)
        self.y_breaks = np.linspace(0.0, ny_cells * cell_size_xy, ny_cells + 1)
        self.z_breaks = np.linspace(0.0, nz_cells * cell_size_z, nz_cells + 1)
        
        self._compute_coarse_factors(nx_cells, ny_cells, nz_cells)
        print(f"✅ Maillage 3D initialisé: {nz_cells}×{ny_cells}×{nx_cells} cellules ({time.time()-timeref:.3f} s)")
        self.init_meshing = True

    def _compute_coarse_factors(self, nx, ny, nz):
        def find_factor(n, max_factor=4):
            for f in range(min(max_factor, n), 0, -1):
                if n % f == 0:
                    return f
            return 1
        self.coarse_factors = [find_factor(nx), find_factor(ny), find_factor(nz)]
        print(f"  Facteurs coarse calculés: {self.coarse_factors[0]}×{self.coarse_factors[1]}×{self.coarse_factors[2]}")

    def load_iaea3d_mat(self):
        """Charge les matériaux IAEA 3D"""
        # F1 - Fuel standard
        self.F1 = {'D': [1.5, 0.4], 'ABS': [0.010, 0.085], 'NSF': [0.0, 0.135], 'CHI': [1., 0.], 'S12': 0.02, 'S21': 0.}
        self.F1['SIGR'] = [self.F1['ABS'][0] + self.F1['S12'], self.F1['ABS'][1]]
        
        # F2 - Control rod (absorbant)
        self.F2 = {'D': [1.5, 0.4], 'ABS': [0.010, 0.130], 'NSF': [0.0, 0.135], 'CHI': [1., 0.], 'S12': 0.02, 'S21': 0.}
        self.F2['SIGR'] = [self.F2['ABS'][0] + self.F2['S12'], self.F2['ABS'][1]]
        
        # F3 - Fuel avec absorption réduite
        self.F3 = {'D': [1.5, 0.4], 'ABS': [0.010, 0.080], 'NSF': [0.0, 0.135], 'CHI': [1., 0.], 'S12': 0.02, 'S21': 0.}
        self.F3['SIGR'] = [self.F3['ABS'][0] + self.F3['S12'], self.F3['ABS'][1]]
        
        # F4 - Réflecteur
        self.F4 = {'D': [2.0, 0.3], 'ABS': [0.000, 0.0100], 'NSF': [0.0, 0.0], 'CHI': [0., 0.], 'S12': 0.04, 'S21': 0.}
        self.F4['SIGR'] = [self.F4['ABS'][0] + self.F4['S12'], self.F4['ABS'][1]]
        
        # F5 - Control rod full insertion
        self.F5 = {'D': [2.0, 0.3], 'ABS': [0.000, 0.0550], 'NSF': [0.0, 0.0], 'CHI': [0., 0.], 'S12': 0.04, 'S21': 0.}
        self.F5['SIGR'] = [self.F5['ABS'][0] + self.F5['S12'], self.F5['ABS'][1]]
        
        # F6 - Vide (absorbant très fort)
        self.F6 = {'D': [0.001, 0.001], 'ABS': [1E15, 1E15], 'NSF': [0.0, 0.0], 'CHI': [0., 0.], 'S12': 0., 'S21': 0.}
        self.F6['SIGR'] = self.F6['ABS'].copy()
        
        self.R0 = self.F6.copy()  # Alias
        print("✅ Matériaux 3D chargés: F1-F5 (fuel/ctrl/refl) + F6/R0 (vide)")

    def init_solver(self):
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé d'abord")
        timeref = time.time()
        print(f"\n=== INITIALISATION SOLVEUR 3D RT{self.rt_order}-P{self.p_order} ===")
        
        if self.rt_order == self.p_order:
            self.mysolv = neutron_solver.NeutFEM(self.rt_order, self.num_groups, self.x_breaks, self.y_breaks, self.z_breaks)
        else:
            self.mysolv = neutron_solver.NeutFEM(self.rt_order, self.p_order, self.num_groups, self.x_breaks, self.y_breaks, self.z_breaks)
        
        self.mysolv.set_linear_solver(LinearSolverType.BICGSTAB)
        
        # Conditions aux limites 3D
        self.mysolv.set_bc(int(BoundaryID.LEFT_3D), BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(int(BoundaryID.RIGHT_3D), BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(int(BoundaryID.TOP_3D), BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(int(BoundaryID.BOTTOM_3D), BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(int(BoundaryID.FRONT_3D), BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(int(BoundaryID.BACK_3D), BCType.DIRICHLET, 0.0)
        
        if "quart" in self.domaine:
            self.mysolv.apply_quarter_rotational_symmetry(0, 1)
            print("  Symétrie quart cyclique activée")
        else:
            print("  Domaine ENTIER : Dirichlet sur tous les bords")

        # Remplissage des coefficients 3D
        Nz, Ny, Nx = self.maillage.shape
        print(f"\n=== REMPLISSAGE 3D ({Nz}×{Ny}×{Nx}) ===")
        
        for k in range(Nz):
            for i in range(Ny):
                for j in range(Nx):
                    fuel_key = self.maillage[k, i, j].strip()
                    mat = getattr(self, fuel_key) if hasattr(self, fuel_key) and fuel_key else self.F6
                    
                    for g in range(self.num_groups):
                        self.mysolv.get_D()[g, k, i, j] = mat['D'][g]
                        self.mysolv.get_NSF()[g, k, i, j] = mat['NSF'][g]
                        self.mysolv.get_Chi()[g, k, i, j] = mat['CHI'][g]
                        self.mysolv.get_SigR()[g, k, i, j] = mat['SIGR'][g]
                    
                    self.mysolv.get_SigS()[1, 0, k, i, j] = mat['S12']
                    self.mysolv.get_SigS()[0, 1, k, i, j] = mat['S21']
        
        self.mysolv.BuildMatrices()
        print(f"\n✅ Solveur 3D initialisé en {time.time()-timeref:.3f} s")

    def solve(self, forward=True, adjoint=False, use_direct_keff=False):
        if self.mysolv is None:
            raise RuntimeError("Le solveur doit être initialisé")
        timeref = time.time()
        self.mysolv.set_tol(1e-5, 1e-4, 1e-4, 200, 1000)
        
        print("\n=== RÉSOLUTION K-EFF 3D ===")
        print(f"  Solveur : {self.mysolv.GetSolverName()}")
        print(f"  Éléments: RT{self.rt_order}-P{self.p_order}")
        
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
            print(f"   k-eff direct    = {self.keff:.6f}")
            print(f"   k-eff référence = {self.kref:.6f}")
            print(f"   Écart absolu    = {ecart_pcm:+.2f} pcm")
        if adjoint:
            print(f"   k-eff adjoint   = {self.keff_adj:.6f}")
        print(f"   Temps résolution = {time1-timeref:.2f} s")
        print("="*60)

        # Calcul puissance intégrée sur z
        if self.p_order == 0 and self.phi is not None:
            Nz, Ny, Nx = self.maillage.shape
            self.pvol = np.zeros((Nz, Ny, Nx))
            for k in range(Nz):
                for i in range(Ny):
                    for j in range(Nx):
                        for g in range(self.num_groups):
                            self.pvol[k, i, j] += self.mysolv.get_NSF()[g, k, i, j] * self.mysolv.get_flux()[g, k, i, j]
            
            # Facteurs assemblages (intégration z)
            self.Fass = self.pvol.reshape((19, self.nmeshes_z, 19, self.nmeshes, 19, self.nmeshes))
            self.Fass = self.Fass.sum(axis=1).sum(axis=2).sum(axis=3)
            self.Fass = 177. * self.Fass / self.Fass.sum()

    def export_vtk(self, filename="iaea3d", export_adjoint=True):
        if self.mysolv is None:
            print("❌ Solveur non initialisé")
            return
        self.mysolv.ExportVTK(filename, export_flux=True, export_current=True, export_xs=True,
                              export_adjoint=export_adjoint and self.phi_adj is not None)
        print(f"✅ Export VTK: {filename}.vtk")

    def plot_flux(self, group=0, plan=10, adjoint=False):
        data = self.phi_adj if adjoint else self.phi
        label = "adjoint" if adjoint else "direct"
        if data is None:
            print(f"❌ Flux {label} non disponible")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[group, plan, :, :], cmap='jet')
        keff = self.keff_adj if adjoint else self.keff
        plt.title(f"Flux Groupe {group+1} ({label}) Plan {plan} - k-eff = {keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_pvol(self, plan=10):
        if self.pvol is None:
            print("❌ Puissance non disponible")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pvol[plan, :, :], cmap='jet')
        plt.title(f"Distribution de puissance Plan {plan} - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_Fass(self):
        if self.Fass is None:
            print("❌ Facteurs assemblages non disponibles")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.Fass, cmap='jet', annot=True, fmt=".2f")
        plt.title(f"Facteurs assemblages intégrés z - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAEA 3D - Solveur neutronique (Version Eigen)")
    parser.add_argument("--mesh", type=str, default="2x2", help="Résolution XY (ex: 2x2, 4x4)")
    parser.add_argument("--mesh-z", type=int, default=1, help="Raffinement axial par plan")
    parser.add_argument("--domain", type=str, default="entier", choices=["entier", "quart_so"])
    parser.add_argument("--rt-order", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--p-order", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--order", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--adjoint_only", action="store_true")
    parser.add_argument("--adjoint", action="store_true")
    parser.add_argument("--use-direct-keff", action="store_true")
    parser.add_argument("--no-coarse", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--vtk", type=str, default=None)
    args = parser.parse_args()

    rt_order = args.order if args.order is not None else args.rt_order
    p_order = args.order if args.order is not None else args.p_order
    solve_forward = not args.adjoint_only
    solve_adjoint = args.adjoint or args.adjoint_only

    print("="*60)
    print("BENCHMARK IAEA 3D - Version Eigen")
    print(f"  → Éléments finis : RT{rt_order}-P{p_order}")
    print("  → Solveur: BiCGSTAB")
    print("="*60)

    iaea3d = Iaea3D(meshtype=args.mesh, nmeshes_z=args.mesh_z, domaine=args.domain)
    iaea3d.rt_order = rt_order
    iaea3d.p_order = p_order
    iaea3d.use_coarse = not args.no_coarse
    iaea3d.load_iaea3d_mat()
    iaea3d.mesh_initialisation()
    iaea3d.init_solver()
    iaea3d.solve(forward=solve_forward, adjoint=solve_adjoint, use_direct_keff=args.use_direct_keff)

    if args.vtk:
        iaea3d.export_vtk(args.vtk, export_adjoint=solve_adjoint)
    if args.plot:
        if solve_forward:
            iaea3d.plot_flux(group=0, plan=10, adjoint=False)
            iaea3d.plot_flux(group=1, plan=10, adjoint=False)
        if solve_adjoint:
            iaea3d.plot_flux(group=0, plan=10, adjoint=True)
        if iaea3d.pvol is not None:
            iaea3d.plot_pvol(plan=10)

    print(f"\n⏱️  Temps total : {time.time() - iaea3d.start:.2f} s")
    print("="*60)
