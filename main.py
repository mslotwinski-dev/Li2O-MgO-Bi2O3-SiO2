import math
import numpy as np
import torch
import argparse
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

parser = argparse.ArgumentParser(
    description="Symulacja MD dla domieszkowanego układu szklistego Li2O-MgO-Bi2O3-SiO2 przy użyciu MACE.")

parser.add_argument(
    '--doping',
    type=str,
    default='LiF',
    help="Materiał domieszkowania."
)
parser.add_argument(
    '--amount',
    type=int,
    help="Ilość moli domieszkowania [0-20).",
    default='0'
)

args = parser.parse_args()

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

def setup_glass_cell(moles_Li2O, moles_MgO, moles_Bi2O3, moles_SiO2, dopant_type=None, moles_dopant=0, target_atoms=1000, density_g_cm3=1.8):
    dopant_stoich = {
        'LiF':  {'Li': 1, 'F': 1},
        'LiCl': {'Li': 1, 'Cl': 1},
        'LiI':  {'Li': 1, 'I': 1},
        'Li2S': {'Li': 2, 'S': 1}
    }

    atoms_dict = {
        'Li': 2 * moles_Li2O,
        'Mg': 1 * moles_MgO,
        'Bi': 2 * moles_Bi2O3,
        'Si': 1 * moles_SiO2,
        'O':  1 * moles_Li2O + 1 * moles_MgO + 3 * moles_Bi2O3 + 2 * moles_SiO2
    }

    if dopant_type and moles_dopant > 0:
        if dopant_type not in dopant_stoich:
            raise ValueError(f"Nieznana domieszka: {dopant_type}. Wybierz coś z {list(dopant_stoich.keys())}")
        
        for elem, count in dopant_stoich[dopant_type].items():
            atoms_dict[elem] = atoms_dict.get(elem, 0) + count * moles_dopant

    total_atoms_moles = sum(atoms_dict.values())
    final_counts = {}
    
    for elem, moles in atoms_dict.items():
        final_counts[elem] = int(np.round((moles / total_atoms_moles) * target_atoms))

    current_total = sum(final_counts.values())
    diff = target_atoms - current_total
    if diff != 0:
        max_elem = max(final_counts, key=final_counts.get)
        final_counts[max_elem] += diff

    symbols = []
    for elem, count in final_counts.items():
        if count > 0:
            symbols += [elem] * count

    np.random.seed(42)
    np.random.shuffle(symbols)

    comp_str = ", ".join([f"{k}:{v}" for k, v in final_counts.items() if v > 0])
    print(f"Skład komórki: {comp_str}")

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

x = args.amount 
typ_domieszki = args.doping 

if 20 < x:
    raise ValueError("Wartość x jest większa niż początkowa ilość Li2O!")

atoms = setup_glass_cell(moles_Li2O=20 - x, moles_MgO=20, moles_Bi2O3=10, moles_SiO2=50, dopant_type=typ_domieszki, moles_dopant=x, target_atoms=target_atoms_count, density_g_cm3=poczatkowa_gestosc)
atoms.write("1_start_loose.xyz")

calculator = mace_mp(model="medium", dispersion=False, default_dtype="float64", device="cuda" if on_cuda else "cpu")
atoms.calc = calculator

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


domieszka = f"{typ_domieszki}-{x}_" if x > 0 else ""
nazwa = f"Li2O_MgO_Bi2O3_SiO2_{domieszka}final.xyz"

atoms.write(nazwa)
print("Sukces! Gotowe.")