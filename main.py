import math
import numpy as np
import torch
from ase import Atoms
from ase.optimize import FIRE, LBFGS
from ase.geometry import get_distances
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.filters import ExpCellFilter
from ase import units
from mace.calculators import mace_mp
from tqdm import tqdm

on_cuda = False

print(f"Czy CUDA jest dostępne? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Karta graficzna: {torch.cuda.get_device_name(0)}")
    on_cuda = True
else:
    print("UWAGA: Brak CUDA! Obliczenia będą szły wolno w chuj.")

def remove_overlaps(atoms, min_dist=1.5, iterations=20):
    for _ in range(iterations):
        pos = atoms.get_positions()
        moved = False

        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                rij = pos[j] - pos[i]
                d = np.linalg.norm(rij)

                if d < min_dist:
                    shift = (min_dist - d) * rij / d * 0.5
                    pos[i] -= shift
                    pos[j] += shift
                    moved = True

        atoms.set_positions(pos)

        if not moved:
            break

def setup_glass_cell(moles_Li2O, moles_MgO, moles_Bi2O3, moles_SiO2, target_atoms=1000, density_g_cm3=1.8):
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

    # Dokładna masa całkowita
    temp_atoms = Atoms(symbols)
    total_mass_amu = temp_atoms.get_masses().sum()

    mass_g = total_mass_amu * 1.660539e-24
    vol_cm3 = mass_g / density_g_cm3
    vol_A3 = vol_cm3 * 1e24
    box_length = vol_A3**(1/3)

    print(f"Długość boku (luźnej) komórki: {box_length:.2f} Å (gęstość: {density_g_cm3} g/cm3)")

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

target_atoms_count = 50
poczatkowa_gestosc = 3.8
docelowa_gestosc = 4.5

atoms = setup_glass_cell(moles_Li2O=20, moles_MgO=20, moles_Bi2O3=10, moles_SiO2=50,
                         target_atoms=target_atoms_count, density_g_cm3=poczatkowa_gestosc)
atoms.write("1_start_loose.xyz")

calculator = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda" if on_cuda else "cpu")
atoms.calc = calculator

# dyn = Langevin(atoms, 0.5*units.fs, temperature_K=5000, friction=0.05)
# dyn.run(500)

print("\nMinimalizacja geometrii początkowej...")
max_opt_steps = 1000
remove_overlaps(atoms)
opt = LBFGS(atoms, maxstep=0.05)
opt.run(fmax=5, steps=2000)
atoms.write("2_minimized_loose.xyz")

forces = atoms.get_forces()
current_fmax = np.max(np.linalg.norm(forces, axis=1))
print(f"Końcowe fmax po optymalizacji: {current_fmax:.2f} eV/A")

if current_fmax > 10.0:
    raise RuntimeError("Geometria nadal bardzo zła (fmax > 10)")

T_melt = 3500
timestep = 1.0 * units.fs

MaxwellBoltzmannDistribution(atoms, temperature_K=T_melt)
Stationary(atoms)
ZeroRotation(atoms)

dyn = Langevin(atoms, timestep, temperature_K=T_melt, friction=0.002)

print(f"\nRozgrzewanie układu do {T_melt} K...")
dyn.run(steps=2000)

print(f"\nKompresja roztopionego szkła do gęstości {docelowa_gestosc} g/cm3...")
mass_g = atoms.get_masses().sum() * 1.660539e-24
target_vol_A3 = (mass_g / docelowa_gestosc) * 1e24
target_L = target_vol_A3**(1/3)

current_L = atoms.get_cell()[0, 0]
cycles_comp = 25
steps_per_comp = 200
L_step = (current_L - target_L) / cycles_comp

for i in tqdm(range(cycles_comp), desc="Kompresja", unit="cykl"):
    new_L = current_L - (i + 1) * L_step
    scale_factor = new_L / atoms.get_cell()[0, 0]
    atoms.set_cell(atoms.get_cell() * scale_factor, scale_atoms=True)
    dyn.run(steps_per_comp)

print("\nRównoważenie stopionego szkła w docelowej gęstości...")
dyn.run(steps=20000)
atoms.write("3_melted_compressed.xyz")

T_final = 300
steps_quench_total = 100000
steps_per_cycle = 200
cycles = steps_quench_total // steps_per_cycle
T_step = (T_melt - T_final) / cycles

print(f"\nChłodzenie układu ({steps_quench_total} kroków w {cycles} cyklach)...")
for i in tqdm(range(cycles), desc="Hartowanie", unit="cykl"):
    current_T = T_melt - (i * T_step)
    dyn.set_temperature(temperature_K=current_T)
    dyn.run(steps_per_cycle)
    if i % 5 == 0:
        Stationary(atoms)

atoms.write("4_quenched.xyz")

print("\nKońcowa relaksacja komórki i atomów...")

box_relax = ExpCellFilter(atoms)
opt_final = FIRE(box_relax, maxmove=0.1)
opt_final.run(fmax=2.5, steps=2000)

final_vol = atoms.get_volume()
final_density = (atoms.get_masses().sum() * 1.660539e-24) / (final_vol * 1e-24)
print(f"\nGęstość po schłodzeniu i końcowej relaksacji: {final_density:.2f} g/cm3")

atoms.write("Li2O-MgO-Bi2O3-SiO2_final.xyz")
print("Sukces! Gotowe.")