# =========================================================================
# MAKEFILE NEUTFEM
# =========================================================================

# =========================================================================
# DÉFINITION DES CHEMINS DES PRÉREQUIS
# =========================================================================
# GCC, pybind11 eigen et python
GCC = XXXX TO COMPLETE XXXX
ANACONDA_VERSION = XXXX TO COMPLETE XXXX
PYBIND = XXXX TO COMPLETE XXXX
EIGEN = XXXX TO COMPLETE XXXX

# =========================================================================
# COMPILATEUR ET DRAPEAUX
# =========================================================================
CXX = $(GCC)/bin/g++-13.2.0

# -fPIC (position independent code) est essentiel pour les bibliothèques partagées (.so)
CXXFLAGS = -fPIC -O3 -std=c++17 -march=native -ffast-math \
           -Wno-deprecated -fvisibility=hidden -finput-charset=UTF-8

# Nom du module de sortie
COXLIB = neutfem/_neutfem_eigen.so

# =========================================================================
# INCLUSIONS ET LIENS
# =========================================================================
# Chemins d'en-têtes (pour la compilation des .o)
INC = -I $(ANACONDA_VERSION)/include/python3.12/ \
      -I $(PYBIND)/include \
      -I $(EIGEN) \
      -I ./include

# Dépendances systèmes critiques pour la liaison partagée
SYS_LIBS = -lrt -lstdc++ -lm -ldl -lpthread

# Chemins des bibliothèques (-L)
LDFLAGS = -L $(ANACONDA_VERSION)/lib

# Bibliothèques à lier (-l)
LDLIBS = $(SYS_LIBS)

# Fichiers sources (objets intermédiaires)
SRC = lib/NeutFEM.o \
      lib/solvers.o \
      lib/FEM.o \
      lib/wrapper.o 

# Répertoires à créer
DIRS = lib neutfem

# =========================================================================
# RÈGLES MAKE
# =========================================================================

# Règle pour créer les répertoires
$(DIRS):
	mkdir -p $@

# Règle principale: dépend de la création des dossiers et du fichier final
all: $(DIRS) $(COXLIB)

# Règle de liaison: dépend des objets
$(COXLIB): $(SRC)
	@echo "Linking shared library $(COXLIB)..."
	$(CXX) -shared -fopenmp -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "✓ Build successful: $(COXLIB)"

# Règle de compilation des objets (.o)
lib/%.o: src/%.cpp include/NeutFEM.hpp
	@echo "Compiling $<..."
	$(CXX) -c $< -o $@ $(INC) -fopenmp $(CXXFLAGS)

# Test de compilation (sans Python)
test: $(DIRS) lib/NeutFEM_Eigen_mesh.o lib/NeutFEM_Eigen_core.o lib/NeutFEM_Eigen_solve.o lib/test_neutfem.o
	@echo "Building test executable..."
	$(CXX) -fopenmp -o test_neutfem $^ $(LDFLAGS) $(LDLIBS)
	@echo "Running tests..."
	./test_neutfem

lib/test_neutfem.o: src/test_neutfem.cpp include/NeutFEM_Eigen.hpp
	@echo "Compiling test..."
	$(CXX) -c $< -o $@ $(INC) -fopenmp $(CXXFLAGS)

# Nettoyage
clean:
	rm -f lib/*.o neutfem/*.so test_neutfem

# Nettoyage complet
distclean: clean
	rm -rf lib neutfem

# Aide
help:
	@echo "NeutFEM Eigen - Makefile"
	@echo ""
	@echo "Cibles disponibles:"
	@echo "  all       - Compile le module Python (défaut)"
	@echo "  test      - Compile et exécute les tests"
	@echo "  clean     - Supprime les fichiers objets et .so"
	@echo "  distclean - Supprime tout (lib/, neutfem/)"
	@echo "  help      - Affiche cette aide"
	@echo ""
	@echo "Dépendances:"
	@echo "  - Eigen3: $(EIGEN)"
	@echo "  - pybind11: $(PYBIND)"
	@echo "  - Python: $(ANACONDA_VERSION)"

.PHONY: all clean distclean test help
