import os
import numpy as np


def generate_jobfile(job_name, core_number, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    text = f"""\
#!/usr/bin/bash

#SBATCH -J {job_name}
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks {core_number}

mpirun -np {core_number} lmp_mpi -in in.self_assemble >& output
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


def generate_lammps_input_file(tem, chi, T_size, box_size_a, box_size_b, box_size_c, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    text = f"""\
#Initialization
variable        tem  equal {format(tem,'.2f')}
variable        pre  equal 0.0
variable        xt   equal {format(chi,'.5f')}
variable        nT   equal {T_size}
variable        nS   equal round(${{nT}}/${{xt}}-${{nT}})
variable        runstep equal 20000000
variable        dumpgap equal ${{runstep}}/1000
variable        thergap equal ${{runstep}}/1000

units           metal
boundary        p p p
atom_style      atomic
newton          on
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# System definition
variable        a    equal {format(box_size_a,'.3f')}
variable        b    equal {format(box_size_b,'.3f')}
variable        c    equal {format(box_size_c,'.3f')}

# Center the boxs
variable        xlo  equal -0.5*${{a}}
variable        xhi  equal  0.5*${{a}}
variable        ylo  equal -0.5*${{b}}
variable        yhi  equal  0.5*${{b}}
variable        zlo  equal -0.5*${{c}}
variable        zhi  equal  0.5*${{c}}

region          box block ${{xlo}} ${{xhi}} ${{ylo}} ${{yhi}} ${{zlo}} ${{zhi}} units box
create_box      2 box
create_atoms    1 random ${{nT}} 123456 box
create_atoms    2 random ${{nS}} 654321 box
mass            1 28.0855
mass            2 28.0855

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

thermo          ${{thergap}}
thermo_style    custom step temp press vol etotal
thermo_modify   lost ignore flush yes

dump            1 all atom ${{dumpgap}} npt.lammpstrj
dump_modify     1 sort id
fix             1 all npt temp ${{tem}} ${{tem}} 2.5 aniso ${{pre}} ${{pre}} 12.5
run             ${{runstep}}

write_data      out.data
"""
    with open(f"{directory}/in.self_assemble", "w") as file:
        file.write(text)


framework = "SOD"
# lattice parameters
a = 8.9561
b = 8.9561
c = 8.9561
super_cell = [8, 8, 8]
box_length_a, box_length_b, box_length_c = a*super_cell[0], b*super_cell[1], c*super_cell[2]
num_T_unit = 12
num_S_unit = 2
chi_T = num_T_unit / (num_T_unit + num_S_unit)
T_size = num_T_unit * super_cell[0] * super_cell[1] * super_cell[2]


solutions = np.genfromtxt("inverse.log", skip_header=1)
criteria = solutions[:, 12] > 480 # zeolite size is larger than 480
solutions = solutions[criteria]
print(f"There are {len(solutions)} solutions that meet the criteria.")

core_number_per_job = 16

for solution in solutions:
    generation, sample, tem, e_TT, e_TS, e_SS, s_TS, s_SS, lam = solution[2:11]
    generate_lammps_input_file(tem, chi_T, T_size, box_length_a, box_length_b, box_length_c, f"generation_{int(generation)}_samples_{int(sample)}")
    generate_TS_file(e_TT, e_TS, e_SS, s_TS, s_SS, lam, f"generation_{int(generation)}_samples_{int(sample)}")
    generate_jobfile(f"sod_{int(generation)}_{int(sample)}", core_number_per_job, f"generation_{int(generation)}_samples_{int(sample)}")
    os.system(f"cd generation_{int(generation)}_samples_{int(sample)} && sbatch job.sh")
