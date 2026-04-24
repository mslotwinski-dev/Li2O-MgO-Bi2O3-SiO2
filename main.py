import math
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.filters import ExpCellFilter
from ase import units
from mace.calculators import mace_mp
from tqdm import tqdm


def setup_glass_cell(moles_Li2O, moles_MgO, moles_Bi2O3, moles_SiO2, target_atoms=1000, density_g_cm3=4.0):
    total_moles = moles_Li2O + moles_MgO + moles_Bi2O3 + moles_SiO2
    n_Li = (2 * moles_Li2O) / total_moles
    n_Mg = (1 * moles_MgO) / total_moles
    n_Bi = (2 * moles_Bi2O3) / total_moles
    n_Si = (1 * moles_SiO2) / total_moles
    n_O = (1 * moles_Li2O + 1 * moles_MgO + 3 * moles_Bi2O3 + 2 * moles_SiO2) / total_moles
    total_fraction = n_Li + n_Mg + n_Bi + n_Si + n_O

    count_Li = int(np.round((n_Li / total_fraction) * target_atoms))
    count_Mg = int(np.round((n_Mg / total_fraction) * target_atoms))
    count_Bi = int(np.round((n_Bi / total_fraction) * target_atoms))
    count_Si = int(np.round((n_Si / total_fraction) * target_atoms))
    count_O = target_atoms - (count_Li + count_Mg + count_Bi + count_Si)

    symbols = ['Li']*count_Li + ['Mg']*count_Mg + ['Bi']*count_Bi + ['Si']*count_Si + ['O']*count_O
    np.random.seed(42)
    np.random.shuffle(symbols)

    print(f"Skład komórki: Li:{count_Li}, Mg:{count_Mg}, Bi:{count_Bi}, Si:{count_Si}, O:{count_O}")

    atomic_masses = {'Li': 6.94, 'Mg': 24.30, 'Bi': 208.98, 'Si': 28.08, 'O': 16.00}
    total_mass_amu = (count_Li * atomic_masses['Li'] + count_Mg * atomic_masses['Mg'] +
                      count_Bi * atomic_masses['Bi'] + count_Si * atomic_masses['Si'] + count_O * atomic_masses['O'])

    mass_g = total_mass_amu * 1.660539e-24
    vol_cm3 = mass_g / density_g_cm3
    vol_A3 = vol_cm3 * 1e24
    box_length = vol_A3**(1/3)

    print(f"Długość boku komórki symulacyjnej: {box_length:.2f} Å")

    grid_size = math.ceil(target_atoms ** (1/3))
    spacing = box_length / grid_size

    positions = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if len(positions) < target_atoms:
                    positions.append([x * spacing, y * spacing, z * spacing])

    atoms = Atoms(symbols=symbols, positions=positions, cell=[box_length, box_length, box_length], pbc=True)
    atoms.rattle(stdev=spacing * 0.2, seed=42)

    return atoms

# ==========================================
# 1. SETUP KOMÓRKI
# ==========================================
atoms = setup_glass_cell(moles_Li2O=20, moles_MgO=20, moles_Bi2O3=10, moles_SiO2=50,
                         target_atoms=1000, density_g_cm3=4.5)
atoms.write("Li2O-MgO-Bi2O3-SiO2_start.xyz")

# ==========================================
# 2. PODPIĘCIE KALKULATORA MACE
# ==========================================
calculator = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
atoms.calc = calculator

# ==========================================
# 3. MINIMALIZACJA ENERGII POCZĄTKOWEJ
# ==========================================
print("\nMinimalizacja geometrii początkowej...")
opt = FIRE(atoms)

max_opt_steps = 1000
pbar_opt1 = tqdm(total=max_opt_steps, desc="Minimalizacja", unit="krok")

def update_pbar1():
    pbar_opt1.update(1)

opt.attach(update_pbar1, interval=1)
# Używamy fmax=0.5 jako łagodnego progu na tym etapie (głównie chcemy usunąć nakładające się atomy)
opt.run(fmax=0.5, steps=max_opt_steps) 
pbar_opt1.close()

atoms.write("Li2O-MgO-Bi2O3-SiO2_min.xyz")

# ==========================================
# 4. TOPIENIE (MD)
# ==========================================
T_melt = 3500
timestep = 0.5 * units.fs
steps_melt = 10000  # Zwiększono lekko dla lepszego upłynnienia

MaxwellBoltzmannDistribution(atoms, temperature_K=T_melt)
Stationary(atoms) # Usuwa pęd środka masy
ZeroRotation(atoms) # Usuwa rotację układu

dyn = Langevin(atoms, timestep, temperature_K=T_melt, friction=0.002)

print(f"\nTopienie szkła w {T_melt} K...")
pbar_melt = tqdm(total=steps_melt, desc="Topienie", unit="krok")

def update_melt():
    pbar_melt.update(10)
    # Co 1000 kroków profilaktycznie usuwamy pęd środka masy (Flying ice cube prevention)
    if pbar_melt.n % 1000 == 0:
        Stationary(atoms)

dyn.attach(update_melt, interval=10)
dyn.run(steps=steps_melt)
pbar_melt.close()

dyn.observers = [] # Czyszczenie obserwatorów

# ==========================================
# 5. HARTOWANIE (QUENCHING)
# ==========================================
T_final = 300
steps_quench_total = 20000 # Lekko wydłużono
steps_per_cycle = 200
cycles = steps_quench_total // steps_per_cycle
T_step = (T_melt - T_final) / cycles

print(f"\nChłodzenie układu ({steps_quench_total} kroków w {cycles} cyklach)...")

for i in tqdm(range(cycles), desc="Hartowanie", unit="cykl"):
    current_T = T_melt - (i * T_step)
    dyn.set_temperature(temperature_K=current_T)
    dyn.run(steps_per_cycle)
    Stationary(atoms) # Utrzymanie zerowego pędu środka masy

# ==========================================
# 6. KOŃCOWA RELAKSACJA Z OPTYMALIZACJĄ KOMÓRKI
# ==========================================
print("\nZapisywanie struktury (końcowa relaksacja komórki i pozycji)...")

# Używamy ExpCellFilter, aby pozwolić na zmianę objętości (pudełko symulacyjne będzie "oddychać")
box_relax = ExpCellFilter(atoms)
opt_final = FIRE(box_relax)

max_final_steps = 2000
pbar_opt2 = tqdm(total=max_final_steps, desc="Relaksacja", unit="krok")

def update_pbar2():
    pbar_opt2.update(1)

opt_final.attach(update_pbar2, interval=1)
# Tutaj optymalizujemy do momentu zrównoważenia sił (fmax=0.05 to dobry standard w ciele stałym)
opt_final.run(fmax=0.15, steps=max_final_steps)
pbar_opt2.close()

# Wypisanie końcowej gęstości
final_vol = atoms.get_volume()
total_mass_amu = atoms.get_masses().sum()
mass_g = total_mass_amu * 1.660539e-24
final_density = mass_g / (final_vol * 1e-24)
print(f"\nGęstość po schłodzeniu i relaksacji: {final_density:.2f} g/cm3")

atoms.write("Li2O-MgO-Bi2O3-SiO2_final.xyz")
print("Gotowe!")