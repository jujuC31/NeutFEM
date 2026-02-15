"""
BENCHMARK BIBLIS 2D - Version Eigen (sans MFEM)
Adaptée pour neutfem_eigen avec:
  - Ordres mixtes RT_k-P_m
  - Solveur adjoint
  - Export VTK
  - Initialisation coarse mesh
  - Solveur BiCGSTAB (sans préconditionneur)
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


class Biblis2D:
    """
    Benchmark BIBLIS 2D avec fonctionnalités avancées
    Version adaptée pour NeutFEM Eigen
    """
    
    def __init__(self, meshtype="2x2", domaine="entier", ncpu=1, sym="cyclique"):
        
        self.start = time.time()
        self.meshtype = meshtype
        self.domaine = domaine
        self.ncpu = ncpu
        self.sym = sym
        
        self.kref = 1.02511  # k-eff référence IAEA
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
        self.coarse_factors = [2, 2, 1]  # Facteurs de réduction par défaut

        # Maillage 17×17 assemblages
        self.maillage_motifs_coeur = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F8", "F1", "F1", "F1", "F1", "F1", "F8", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F5", "F1", "F7", "F1", "F7", "F1", "F7", "F1", "F5", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F5", "F2", "F8", "F2", "F8", "F1", "F8", "F2", "F8", "F2", "F5", "F4", "  ", "  "],
            ["  ", "F4", "F8", "F1", "F8", "F2", "F8", "F2", "F6", "F2", "F8", "F2", "F8", "F1", "F8", "F4", "  "],
            ["  ", "F4", "F1", "F7", "F2", "F8", "F1", "F8", "F2", "F8", "F1", "F8", "F2", "F7", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F1", "F8", "F2", "F8", "F1", "F8", "F1", "F8", "F2", "F8", "F1", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F7", "F1", "F6", "F2", "F8", "F1", "F8", "F2", "F6", "F1", "F7", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F1", "F8", "F2", "F8", "F1", "F8", "F1", "F8", "F2", "F8", "F1", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F7", "F2", "F8", "F1", "F8", "F2", "F8", "F1", "F8", "F2", "F7", "F1", "F4", "  "],
            ["  ", "F4", "F8", "F1", "F8", "F2", "F8", "F2", "F6", "F2", "F8", "F2", "F8", "F1", "F8", "F4", "  "],
            ["  ", "  ", "F4", "F5", "F2", "F8", "F2", "F8", "F1", "F8", "F2", "F8", "F2", "F5", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F5", "F1", "F7", "F1", "F7", "F1", "F7", "F1", "F5", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F8", "F1", "F1", "F1", "F1", "F1", "F8", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
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
        plt.title(f"Géométrie - {self.meshtype} - {self.domaine}")
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
        plt.title("Distribution des matériaux Biblis 2D")
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

        # Coordonnées
        cell_size = 23.1226 / self.nmeshes
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
        # Cherche le plus grand diviseur commun <= 4
        def find_factor(n, max_factor=4):
            for f in range(min(max_factor, n), 0, -1):
                if n % f == 0:
                    return f
            return 1
        
        rx = find_factor(nx)
        ry = find_factor(ny)
        self.coarse_factors = [rx, ry, 1]
        print(f"  Facteurs coarse calculés: {rx}×{ry}×1")

    def load_biblis2d_mat(self, include_upscattering=False):
        """Charge les matériaux avec upscattering optionnel"""
        
        # Matériaux F1-F8
        self.F1 = {
            'D': [1.4360, 0.3635],
            'ABS': [0.0095042, 0.0750580],
            'NSF': [0.0058708, 0.0960670],
            'CHI': [1., 0.],
            'S12': 0.017754
        }
        self.F1['SIGR'] = [self.F1['ABS'][0] + self.F1['S12'], self.F1['ABS'][1]]

        self.F2 = {
            'D': [1.4366, 0.3636],
            'ABS': [0.0096785, 0.0784360],
            'NSF': [0.0061908, 0.1035800],
            'CHI': [1., 0.],
            'S12': 0.017621
        }
        self.F2['SIGR'] = [self.F2['ABS'][0] + self.F2['S12'], self.F2['ABS'][1]]

        self.F4 = {
            'D': [1.4389, 0.3638],
            'ABS': [0.0103630, 0.0914080],
            'NSF': [0.0074527, 0.1323600],
            'CHI': [1., 0.],
            'S12': 0.017101
        }
        self.F4['SIGR'] = [self.F4['ABS'][0] + self.F4['S12'], self.F4['ABS'][1]]

        self.F5 = {
            'D': [1.4381, 0.3665],
            'ABS': [0.0100030, 0.0848280],
            'NSF': [0.0061908, 0.1035800],
            'CHI': [1., 0.],
            'S12': 0.01729
        }
        self.F5['SIGR'] = [self.F5['ABS'][0] + self.F5['S12'], self.F5['ABS'][1]]

        self.F6 = {
            'D': [1.4385, 0.3665],
            'ABS': [0.0101320, 0.0873140],
            'NSF': [0.0064285, 0.1091100],
            'CHI': [1., 0.],
            'S12': 0.017192
        }
        self.F6['SIGR'] = [self.F6['ABS'][0] + self.F6['S12'], self.F6['ABS'][1]]

        self.F7 = {
            'D': [1.4389, 0.3679],
            'ABS': [0.0101650, 0.0880240],
            'NSF': [0.0061908, 0.1035800],
            'CHI': [1., 0.],
            'S12': 0.017125
        }
        self.F7['SIGR'] = [self.F7['ABS'][0] + self.F7['S12'], self.F7['ABS'][1]]

        self.F8 = {
            'D': [1.4393, 0.3680],
            'ABS': [0.0102940, 0.0905100],
            'NSF': [0.0064285, 0.1091100],
            'CHI': [1., 0.],
            'S12': 0.017027
        }
        self.F8['SIGR'] = [self.F8['ABS'][0] + self.F8['S12'], self.F8['ABS'][1]]

        # Réflecteur
        self.R0 = {
            'D': [1.3200, 0.2772],
            'ABS': [0.0026562, 0.0715960],
            'NSF': [0.0000000, 0.0000000],
            'CHI': [0., 0.],
            'S12': 0.023106
        }
        self.R0['SIGR'] = [self.R0['ABS'][0] + self.R0['S12'], self.R0['ABS'][1]]
        
        # ✅ Upscattering optionnel
        if include_upscattering:
            print("\n⚠️  ATTENTION : Ajout d'upscattering (NON standard)")
            upscatter_ratio = 0.08
            
            for fuel_name in ['F1', 'F2', 'F4', 'F5', 'F6', 'F7', 'F8', 'R0']:
                fuel = getattr(self, fuel_name)
                fuel['S21'] = fuel['S12'] * upscatter_ratio
                fuel['SIGR'][1] = fuel['ABS'][1] + fuel['S21']
                print(f"  {fuel_name}: Σₛ(1→0)={fuel['S21']:.6f}")
        else:
            for fuel_name in ['F1', 'F2', 'F4', 'F5', 'F6', 'F7', 'F8', 'R0']:
                getattr(self, fuel_name)['S21'] = 0.0
        
        print("✅ Matériaux chargés: F1-F8 + R0")


    def init_solver(self):
        """Initialise le solveur NeutFEM"""
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé d'abord")
        
        timeref = time.time()
        
        order_str = f"RT{self.rt_order}-P{self.p_order}"
        print(f"\n=== INITIALISATION SOLVEUR {order_str} ===")
        
        # Création du solveur avec ordres distincts
        if self.rt_order == self.p_order:
            # Constructeur classique
            self.mysolv = neutron_solver.NeutFEM(
                self.rt_order,
                self.num_groups,
                self.x_breaks,
                self.y_breaks,
                self.z_breaks
            )
        else:
            # Constructeur avec ordres mixtes
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
                
                self.mysolv.get_SigS()[1, 0, i, j] = mat['S12']
                self.mysolv.get_SigS()[0, 1, i, j] = mat['S21']
        
        # Validation
        print("\n=== VALIDATION DES COEFFICIENTS ===")
        for g in range(self.num_groups):
            D = self.mysolv.get_D()[g]
            print(f"Groupe {g}: D ∈ [{D.min():.6f}, {D.max():.6f}] cm")
        print(f"Scattering 0→1: [{self.mysolv.get_SigS()[1, 0].min():.6f}, {self.mysolv.get_SigS()[1, 0].max():.6f}]")
        print(f"Scattering 1→0: [{self.mysolv.get_SigS()[0, 1].min():.6f}, {self.mysolv.get_SigS()[0, 1].max():.6f}]")
        
        # Construction des matrices
        self.mysolv.BuildMatrices()

        time1 = time.time()
        print(f"\n✅ Solveur initialisé en {time1-timeref:.3f} s")

    def solve(self, forward=True, adjoint=False, use_direct_keff=False):
        """Résout le problème avec initialisation coarse mesh
        
        Args:
            forward: Résoudre le problème direct
            adjoint: Résoudre le problème adjoint
            use_direct_keff: Utiliser k-eff direct pour l'adjoint
        """
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
        
        # Résolution directe
        if forward:
            print("\n--- Problème DIRECT ---")
            self.keff = self.mysolv.SolveKeff(
                use_coarse_init=self.use_coarse,
                coarse_factors=self.coarse_factors
            )
            
            # Récupération du flux
            if self.p_order == 0:
                self.phi = np.array([
                    self.mysolv.get_flux()[g] for g in range(self.num_groups)
                ])

        # Résolution adjointe
        if adjoint:
            print("\n--- Problème ADJOINT ---")
            self.keff_adj = self.mysolv.SolveAdjoint(
                normalize_to_direct=forward,  # Normaliser si direct calculé
                use_direct_keff=use_direct_keff
            )
            
            # Récupération du flux adjoint
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

    def export_vtk(self, filename="biblis2d", export_adjoint=True):
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


# ===== SCRIPT PRINCIPAL =====
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="BIBLIS 2D - Solveur neutronique (Version Eigen + Ordres mixtes)"
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
    parser.add_argument("--upscatter", action="store_true",
                       help="Activer l'upscattering")
    parser.add_argument("--no-coarse", action="store_true",
                       help="Désactiver l'initialisation coarse")
    parser.add_argument("--coarse-factor", type=int, default=2,
                       help="Facteur de réduction coarse (défaut: 2)")
    
    # Sorties
    parser.add_argument("--plot", action="store_true",
                       help="Afficher les graphiques")
    parser.add_argument("--vtk", type=str, default=None,
                       help="Exporter au format VTK (spécifier le nom de fichier)")
    
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
    print("BENCHMARK BIBLIS 2D - Version Eigen")
    print(f"  → Éléments finis : {order_str}")
    print("  → Solveur: BiCGSTAB (sans préconditionneur)")
    print("  → Initialisation: Coarse mesh")
    print("="*60)
    print(f"Maillage  : {args.mesh}")
    print(f"Domaine   : {args.domain}")
    print(f"Coarse    : {'Non' if args.no_coarse else f'Oui (facteur {args.coarse_factor})'}")
    print(f"Upscatter : {'Oui' if args.upscatter else 'Non'}")
    print(f"Direct    : {'Oui' if solve_forward else 'Non'}")
    print(f"Adjoint   : {'Oui' if solve_adjoint else 'Non'}")
    if solve_adjoint:
        print(f"  → k-eff direct : {'Oui' if args.use_direct_keff else 'Non'}")
    print("="*60)
    
    # Création
    biblis2d = Biblis2D(meshtype=args.mesh, domaine=args.domain)
    biblis2d.rt_order = rt_order
    biblis2d.p_order = p_order
    biblis2d.use_coarse = not args.no_coarse
    
    # Matériaux et maillage
    biblis2d.load_biblis2d_mat(include_upscattering=args.upscatter)
    biblis2d.mesh_initialisation()
    
    # Override coarse factors si spécifié
    if args.coarse_factor != 2:
        nx = biblis2d.maillage.shape[1]
        ny = biblis2d.maillage.shape[0]
        if nx % args.coarse_factor == 0 and ny % args.coarse_factor == 0:
            biblis2d.coarse_factors = [args.coarse_factor, args.coarse_factor, 1]
            print(f"  Facteurs coarse override: {biblis2d.coarse_factors}")
        else:
            print(f"  ⚠️  Facteur {args.coarse_factor} ne divise pas {nx}×{ny}, utilisation auto")
    
    # Résolution
    biblis2d.init_solver()
    biblis2d.solve(
        forward=solve_forward,
        adjoint=solve_adjoint,
        use_direct_keff=args.use_direct_keff
    )
    
    # Export VTK
    if args.vtk:
        biblis2d.export_vtk(args.vtk, export_adjoint=solve_adjoint)
    
    # Visualisation
    if args.plot:
        if solve_forward:
            biblis2d.plot_flux(group=0, adjoint=False)
            biblis2d.plot_flux(group=1, adjoint=False)
        if solve_adjoint:
            biblis2d.plot_flux(group=0, adjoint=True)
            biblis2d.plot_flux(group=1, adjoint=True)
        if biblis2d.pvol is not None:
            biblis2d.plot_pvol()

    print(f"\n⏱️  Temps total : {time.time() - biblis2d.start:.2f} s")
    print("="*60)
