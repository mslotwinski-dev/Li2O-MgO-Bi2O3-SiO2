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

target_atoms_count = 1000
poczatkowa_gestosc = 0.2
docelowa_gestosc = 4.5

x = args.amount 
typ_domieszki = args.doping 

if 20 < x:
    raise ValueError("Wartość x jest większa niż początkowa ilość Li2O!")

atoms = setup_glass_cell(moles_Li2O=20 - x, moles_MgO=20, moles_Bi2O3=10, moles_SiO2=50, dopant_type=typ_domieszki, moles_dopant=x, target_atoms=target_atoms_count, density_g_cm3=poczatkowa_gestosc)
calculator = mace_mp(model="medium", dispersion=False, default_dtype="float64", device="cuda" if on_cuda else "cpu")
atoms.calc = calculator

print("\n--- Rozpoczynam skanowanie gęstości układu ---")
print(f"{'Gęstość (g/cm³)':<18} | {'Objętość (Å³)':<18} | {'Energia (eV)':<18}")
print("-" * 60)

total_mass_amu = atoms.get_masses().sum()
mass_g = total_mass_amu * 1.660539e-24
punkty_pomiarowe = 30
gestosci_do_sprawdzenia = np.linspace(poczatkowa_gestosc, docelowa_gestosc, num=punkty_pomiarowe)

wyniki = []
for gestosc in gestosci_do_sprawdzenia:
    vol_cm3 = mass_g / gestosc
    vol_A3 = vol_cm3 * 1e24
    
    new_box_length = vol_A3**(1/3)
    
    atoms.set_cell([new_box_length, new_box_length, new_box_length], scale_atoms=True)
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=0.05, steps=2500)
    energia = atoms.get_potential_energy()
    
    wyniki.append((gestosc, vol_A3, energia))
    print(f"{gestosc:<18.4f} | {vol_A3:<18.2f} | {energia:<18.4f}")

nazwa_pliku = f"energia_od_gestosci_{typ_domieszki}_{x}mol.txt"
with open(nazwa_pliku, "w") as f:
    f.write("Gestosc_g_cm3 Objetosc_A3 Energia_eV\n")
    for g, v, e in wyniki:
        f.write(f"{g:.4f} {v:.2f} {e:.4f}\n")

print(f"\nSkanowanie zakończone. Wyniki zapisano do pliku: {nazwa_pliku}")