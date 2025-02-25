import os
import time
import pickle
import logging
import shutil
import numpy as np
from cma import CMAEvolutionStrategy


def generate_jobfile(job_name, core_number, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    text = f"""\
#!/usr/bin/bash

#SBATCH -J {job_name}
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks {core_number}

mpirun -np {core_number} lmp_mpi -in in.inverse >& output
"""
    with open(f"{directory}/job.sh", "w") as file:
        file.write(text)


def generate_TS_file(e_TT, e_TS, e_SS, s_TS, s_SS, lam, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    text = f"""\
#     epsilon  sigma  a  lambda gamma costheta0     A           B            p q tol
T T T {format(e_TT,'.6f')} 2.7275 1.8 {format(lam,'.2f')} 1.2 -0.333333333333 7.049556277 0.6022245584 4 0 0
S S S {format(e_SS,'.6f')} {format(s_SS,'.2f')}   1.8 0     1.2 -0.333333333333 7.049556277 0.6022245584 4 0 0
T T S 0        0      0   0     0   -0.333333333333 0           0            0 0 0
T S S {format(e_TS,'.6f')} {format(s_TS,'.2f')}   1.8 0     1.2 -0.333333333333 7.049556277 0.6022245584 4 0 0
S T T {format(e_TS,'.6f')} {format(s_TS,'.2f')}   1.8 0     1.2 -0.333333333333 7.049556277 0.6022245584 4 0 0
S T S 0        0      0   0     0   -0.333333333333 0           0            0 0 0
S S T 0        0      0   0     0   -0.333333333333 0           0            0 0 0
T S T 0        0      0   0     0   -0.333333333333 0           0            0 0 0
"""
    with open(f"{directory}/TS.sw", "w") as file:
        file.write(text)


def generate_lammps_input_file(Tem, chi, T_size, seed_size, box_size_a, box_size_b, box_size_c, seed_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    text = f"""\
#Initialization
variable        tem  equal {format(Tem,'.2f')}
variable        pre  equal 0.0
variable        xt   equal {format(chi,'.5f')}
variable        nT   equal {T_size}
variable        nS   equal round(${{nT}}/${{xt}}-${{nT}})
variable        sT   equal {seed_size}
variable        nTn  equal ${{nT}}-${{sT}}

units           metal
boundary        p p p
atom_style      atomic
newton          on
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# System definition
variable        a    equal {box_size_a}
variable        b    equal {box_size_b}
variable        c    equal {box_size_c}

# Center the boxs
variable        xlo  equal -0.5*${{a}}
variable        xhi  equal  0.5*${{a}}
variable        ylo  equal -0.5*${{b}}
variable        yhi  equal  0.5*${{b}}
variable        zlo  equal -0.5*${{c}}
variable        zhi  equal  0.5*${{c}}

region          box block ${{xlo}} ${{xhi}} ${{ylo}} ${{yhi}} ${{zlo}} ${{zhi}} units box
create_box      2 box
read_data       {seed_file} add merge
create_atoms    1 random ${{nTn}} 123456 box
create_atoms    2 random ${{nS}} 654321 box
mass            1 28.0855
mass            2 28.0855

group           seed id <= ${{sT}}
fix             freeze seed setforce 0.0 0.0 0.0

# Minimization
pair_style      sw
pair_coeff      * * TS.sw T S

# Run a simulation
thermo          100
thermo_style    custom step etotal
minimize        1.0e-4 1.0e-6 100 1000   

reset_timestep  0
timestep        0.005

velocity        all create ${{tem}} 123456 rot yes dist gaussian

thermo          2000
thermo_style    custom step temp press vol etotal
thermo_modify   lost ignore flush yes

dump            1 all atom 200 npt.lammpstrj
dump_modify     1 sort id
unfix           freeze
fix             plb all plumed plumedfile plumed.dat outfile log.plumed
fix             1 all npt temp ${{tem}} ${{tem}} 2.5 iso ${{pre}} ${{pre}} 12.5
run             200000

write_data      out.data
"""
    with open(f"{directory}/in.inverse", "w") as file:
        file.write(text)


def generate_plumed_file(seed_size, num_envs, kappa, system_size, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    references = "\n ".join(f"REFERENCE_{i}=../../envs/env{i}.pdb" for i in range(1, num_envs+1))
    text = f"""\
UNITS LENGTH=A ENERGY=eV TIME=ps

ENVIRONMENTSIMILARITY ...
 LABEL=es1
 SPECIES=1-{seed_size}
 SIGMA=0.5
 CRYSTAL_STRUCTURE=CUSTOM
 {references}
 MORE_THAN={{RATIONAL R_0=0.5 NN=6 MM=12}}
...

MOVINGRESTRAINT ...
 LABEL=steer
 ARG=es1.morethan
 VERSE=L
 STEP0=00000 AT0={seed_size} KAPPA0={format(kappa,'.3f')}
 STEP1=200000 AT1={seed_size} KAPPA1={format(kappa,'.3f')}
...

ENVIRONMENTSIMILARITY ...
 LABEL=es2
 SPECIES=1-{system_size}
 SIGMA=0.5
 CRYSTAL_STRUCTURE=CUSTOM
 {references}
 MORE_THAN={{RATIONAL R_0=0.5 NN=6 MM=12}}
...

DUMPMULTICOLVAR DATA=es2 FILE=MULTICOLVAR.xyz STRIDE=2000

FLUSH STRIDE=2000
PRINT ARG=* STRIDE=2000 FILE=COLVAR
"""
    with open(f"{directory}/plumed.dat", "w") as file:
        file.write(text)


def get_num_envs(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def get_seed_size(seed_file, directory):
    file_path = os.path.join(directory, seed_file)
    with open(file_path, "r") as file:
        for line in file:
            if line.strip().endswith('atoms'):
                return int(line.split()[0])
        return "Error: No atoms found in seed file"


def get_fitness(directory):
    zeolite = np.genfromtxt(directory+"/COLVAR")[:, 8]
    zeolite_size = np.mean(zeolite[-25:])  # average of the last 25 values
    fitness = -1 * zeolite_size/T_size
    return zeolite_size, fitness



framework = "SOD"
# lattice parameters, you can find these values in database.
a = 8.9561
b = 8.9561
c = 8.9561
super_cell = [4, 4, 4]
box_length_a, box_length_b, box_length_c = a * super_cell[0], b * super_cell[1], c * super_cell[2]
num_T_unit = 12
num_S_unit = 2
chi_T = num_T_unit / (num_T_unit + num_S_unit)
T_size = num_T_unit * super_cell[0] * super_cell[1] * super_cell[2]

current_dir = os.getcwd()
seed_file = "sod_seed_23.data"
seed_size = get_seed_size(seed_file, current_dir)
bias_strength = 20 # kbT
num_envs = get_num_envs(current_dir+"/envs")
num_cores = 8
num_generations = 50

print(f"Current directory is: {current_dir}")
print(f"The {framework} seed file is: {seed_file}")
print(f"The {framework} seed size is: {seed_size}")
print(f"The number of environments of {framework} is: {num_envs}")
print(f"The number of cores used per sample is: {num_cores}")


# design parameters, T, e_TT, e_TS, e_SS, s_TS, s_SS, lam or chi
design_parameters = {
    "Tem":      [600, 700],     # kelvin
    "e_TT":     [0.3, 0.7],     # ev, 0.536761 in Molinero's solution
    "e_TS":     [0.05, 0.09],   # ev, 0.06938 in Molinero's solution
    "e_SS":     [0.02, 0.09],   # ev, 0.029488 in Molinero's solution
    "s_TS":     [4, 7],         # A, 5.13 in Molinero's solution
    "s_SS":     [6, 10],        # A, 5.13 in Molinero's solution
    "lam":      [15, 25],       # dimensionless, 23.15 in Molinero's solution
    "chi":      chi_T           # dimensionless
}

# check if adding chi as a design parameter
if isinstance(design_parameters["chi"], list):
    num_parameters = len(design_parameters)
else:
    num_parameters = len(design_parameters) - 1

# initialize the optimizer
optimizer = CMAEvolutionStrategy([0.5]*num_parameters, 0.16, {'bounds': [0.0, 1.0]})
logging.basicConfig(filename='inverse.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info("# generation, sample, tem, e_tt, e_ts, e_ss, s_ts, s_ss, lam, chi, zeolite_size, fitness")

# start the iterations, 50 in this case
for generation in range(1, num_generations + 1):
    generation_dir = os.path.join(current_dir, "generation"+str(generation))
    os.makedirs(generation_dir)
    solutions = optimizer.ask()
    real_solutions = []

    # generate the input files for each sample and run the simulations
    for sample, solution in enumerate(solutions):
        # rescale the design parameters according to the design parameter ranges
        rescaled_solution = []
        for i, (key, value) in enumerate(list(design_parameters.items())[:num_parameters]):
            min_val, max_val = value
            rescaled_value = solution[i] * (max_val - min_val) + min_val
            rescaled_solution.append(rescaled_value)
        
        if num_parameters < len(design_parameters):
            rescaled_solution.append(design_parameters["chi"])
        
        real_solutions.append(rescaled_solution)
        Tem, e_TT, e_TS, e_SS, s_TS, s_SS, lam, chi = rescaled_solution

        # create a directory for the current sample
        sample_dir = os.path.join(generation_dir, "sample"+str(sample))
        os.makedirs(sample_dir)
        shutil.copy(seed_file, sample_dir)
        generate_jobfile(f"G{generation}S{sample}", num_cores, sample_dir)
        generate_TS_file(e_TT, e_TS, e_SS, s_TS, s_SS, lam, sample_dir)
        generate_lammps_input_file(Tem, chi, T_size, seed_size, box_length_a, box_length_b, box_length_c, seed_file, sample_dir)
        kappa = 0.00008617 * Tem * bias_strength # kbT -> eV
        generate_plumed_file(seed_size, num_envs, kappa, T_size, sample_dir)
        os.system("cd generation"+str(generation)+"/sample"+str(sample)+" && sbatch job.sh")

    # wait for the simulations in current generation to complete
    done = False
    while (not done):
        time.sleep(10)
        checks = [False]*optimizer.popsize
        for sample in range(optimizer.popsize):
            sample_dir = os.path.join(generation_dir, "sample"+str(sample))
            checks[sample] = os.path.exists(sample_dir+"/out.data")
            done = all(checks)

    # get the fitness of each sample
    fitnesses = []
    for sample in range(optimizer.popsize):
        sample_dir = os.path.join(generation_dir, "sample"+str(sample))
        zeolite_size, fitness = get_fitness(sample_dir)
        fitnesses.append(fitness)
        logging.info(f"{generation}  {sample}  {real_solutions[sample][0]:.2f}  {real_solutions[sample][1]:.6f}  {real_solutions[sample][2]:.6f}  {real_solutions[sample][3]:.6f}  {real_solutions[sample][4]:.2f}  {real_solutions[sample][5]:.2f}  {real_solutions[sample][6]:.2f}  {real_solutions[sample][7]:.2f}  {zeolite_size:.2f}  {fitness:.6f}")

    # feed the fitnesses back to the optimizer
    optimizer.tell(solutions, fitnesses)
    if generation != 0 and generation % 5 == 0:
        with open("optimizer"+str(generation)+".pickled", "wb") as f:
            pickle.dump(optimizer, f)
