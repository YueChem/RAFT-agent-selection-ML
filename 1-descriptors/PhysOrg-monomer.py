#!/usr/bin/env python3
# 20250219
# Modified by Yue Mu

import datetime
import os
import re
import subprocess
import time
import shlex
import pandas as pd
import math



def run_gaussian(input_file, output_file):
    command = f"g09 {input_file} {output_file}"
    print(f"Running: {command}")
    args = shlex.split(command)
    with open(output_file, "a") as mystdout:
        process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            print(f"Error in Gaussian calculation: {err.decode('utf-8')}")
        else:
            print(f"Gaussian calculation finished successfully!")

def check_and_rename_cdft(target_file):
    """check and rename CDFT.txt file"""
    cdft_file = "CDFT.txt"
    if os.path.exists(cdft_file):
        new_name = f"{os.path.splitext(target_file)[0]}_CDFT.txt"
        os.rename(cdft_file, new_name)
        print(f"\nRenamed {cdft_file} to {os.path.basename(new_name)}")
    else:
        print(f"\nNo {cdft_file} file generated for {os.path.basename(target_file)}")

def call_Multiwfn(all_target_files):
    result_file_list = []
    total_num_files = len(all_target_files)
    assert total_num_files > 0, "No target file was found!"
    print(f'Total {total_num_files} \'*{os.path.splitext(all_target_files[0])[1]}\' files were found.')
    # pre_generation
    input_stream_command_pre = [22, 1, "m062x/6-311++g(2d,p)", "0 2", "-1 1", "1 1", 0, 'q']
    input_stream_pre = "\n".join(list(map(str, input_stream_command_pre)))
    # descriptors calculation
    input_stream_command = [0, 9, 3, 'n', 0, 22, 2, 0, 'q']
    input_stream = "\n".join(list(map(str, input_stream_command)))

    for i, target_file in enumerate(all_target_files, start=1):
        result_file = os.path.splitext(target_file)[0] + ".txt"

        print(f'\nUsing Multiwfn to calculate orbitals, charges, and bond order descriptors for {os.path.basename(target_file)}')
        # Generate the Gaussian input file (.gjf) required for calculating charge descriptors
        print('\nGenerating the Gaussian input file (.gjf) required for calculating charge descriptors...')
        arg_pre = f'Multiwfn_noGUI {target_file} << EOF\n{input_stream_pre}\nEOF\n'
        subprocess.run(arg_pre, shell=True)

        # Run the Gaussian calculations
        print(f'\nRun Gaussian to calculate these .gjf files')
        run_gaussian("N.gjf", "N.log")
        run_gaussian("N+1.gjf", "N+1.log")
        run_gaussian("N-1.gjf", "N-1.log")

        # Use Multiwfn to calculate Molecular orbitals, charges, and bond order descriptors
        print(f'\nUsing Multiwfn to calculate orbitals, charges, and bond order descriptors')

        arg = f'Multiwfn_noGUI {target_file} << EOF > {result_file}\n{input_stream}\nEOF\n'
        subprocess.run(arg, shell=True)

        # rename CDFT.txt
        check_and_rename_cdft(target_file)
        cdft_file = f"{os.path.splitext(target_file)[0]}_CDFT.txt"
        with open(cdft_file, 'r') as cdft, open(result_file, 'a') as result:
            result.write("\n")
            result.write(cdft.read())
        print(f"\nAppended {os.path.basename(cdft_file)} to {os.path.basename(result_file)}")

        result_file_list.append(result_file)
    print('\n')

    return result_file_list

def call_morfeus(all_target_file):
    result_file_list = []
    for target_file in all_target_file:
        xyz_base = os.path.splitext(target_file)[0]
        xyz_file = f"{xyz_base}.xyz"
        result_file = f"{xyz_base}_morfeus.txt"

        print(f'\nUsing morfeus to calculate buried volume and dispersion descriptors for {os.path.basename(xyz_file)}')

        # Use Morfeus to calculate buried volume descriptors
        # Parameters definition
        commands = [
            (atom, radius)
            for atom in [2]  # index of radical carbon atom
            for radius in [3.0, 4.0, 5.0]  # sphere radius
        ]
        for i, (atom, radius) in enumerate(commands):
            redirect = '>' if i == 0 else '>>'
            arg = f'morfeus buried_volume {xyz_file} - {atom} --radius={radius} - print_report {redirect} {result_file}'
            subprocess.run(arg, shell=True)

        # Use Morfeus to calculate dispersion descriptors
        # Parameters definition
        dispersion_params = [
            ("- - p_int",),
            ("- - p_max",),
            ("- - p_min",),
            ("- - atom_p_int",)
        ]
        for params in dispersion_params:
            arg = f'morfeus dispersion {xyz_file} {" ".join(params)} >> {result_file}'
            subprocess.run(arg, shell=True)
        result_file_list.append(result_file)
    print('\n')

    return result_file_list


def search(ini_path, target_file_name):
    target_list = []
    for root, dirs, files in os.walk(ini_path, topdown=True):
        for f in files:
            if re.search(target_file_name, f):
                full_path = os.path.join(root, f)
                target_list.append(full_path)
    return target_list

def data_extraction(file_list_Multiwfn,file_list_morfeus):
    data = []
    for file_Multiwfn in file_list_Multiwfn:
        with open(file_Multiwfn, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        c_bond = []
        sample_name = os.path.basename(file_Multiwfn)
        sample_name = os.path.splitext(sample_name)[0]

        for i, line in enumerate(lines):
            if 'Orbital' in line and 'alpha-HOMO' in line:
                somo = float(re.search(r'energy:\s+([\d.-]+)', line).group(1))
            elif 'First vertical IP:' in line:
                IE = float(re.search(r'First vertical IP:\s*(-?\d+\.\d+)', line).group(1))
            elif 'First vertical EA:' in line:
                EA = float(re.search(r'First vertical EA:\s*(-?\d+\.\d+)', line).group(1))

            elif ' 2(' in line and '#' in line and 'Alpha:' not in line:
                match = re.search(r'\s+([\d.]+)$', line)
                if match:
                    c_bond.append(float(match.group(1)))
                c_bond_clean = [value for value in c_bond if value >= 0.5]
                C_avg = sum(c_bond_clean)/len(c_bond_clean) if c_bond_clean else 0.0
                C_max = max(c_bond_clean) if c_bond_clean else 0.0
                C_min = min(c_bond_clean) if c_bond_clean else 0.0

            elif 'condensed Fukui functions' in line:
                if i + 4 < len(lines):
                    C_line = lines[i + 4].strip()
                    columns = C_line.split()
                    C_qN = float(columns[2])
                    C_f_0 = float(columns[7])

            elif 'Condensed local softness' in line:
                if i + 3 < len(lines):
                    C_line = lines[i + 3].strip()
                    columns = C_line.split()
                    C_s_0 = float(columns[4])

        data.append([sample_name, somo, IE, EA,
                     C_max, C_min, C_avg,
                     C_qN, C_f_0, C_s_0,])

    df_Multiwfn = pd.DataFrame(data, columns=['SampleName', 'E1', 'E2', 'E3',
                                              'B1', 'B2', 'B3',
                                              'Q1', 'Q2', 'Q3'])
    df_Multiwfn['E4'] = (df_Multiwfn['E2'] + df_Multiwfn['E3']) / 2
    df_Multiwfn['E5'] = 2 / (df_Multiwfn['E2'] - df_Multiwfn['E3'])

    data = []
    for file_morfeus in file_list_morfeus:
        with open(file_morfeus, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        Buried_volume = []
        for line in lines[:3]:
            parts = line.split()
            if not parts:
                continue
            try:
                value = float(parts[-1])
                Buried_volume.append(value)
            except:
                print(f"Invalid data in line {line_num}: {line}")
        P_int = []
        for line in lines[3:6]:
            parts = line.split()
            if not parts:
                continue
            try:
                value = float(parts[-1])
                P_int.append(value)
            except:
                print(f"Invalid data in line {line_num}: {line}")

        Atom_p_int = []
        for line in lines[6:]:
            parts = line.split()
            if not parts:
                continue
            try:
                num = float(parts[-1])
                Atom_p_int.append(num)
            except:
                print(f"Invalid data in line {line_num}: {line}")

        if len(Atom_p_int) < 2:
            P_std_dev = 0.0
        else:
            mean = sum(Atom_p_int) / len(Atom_p_int)
            variance = sum((x - mean) ** 2 for x in Atom_p_int) / (len(Atom_p_int) - 1)
            P_std_dev = math.sqrt(variance)

        data.append(Buried_volume + P_int + [P_std_dev])

    df_morfeus = pd.DataFrame(data, columns=['V1', 'V2', 'V3',
                                             'P1', 'P2', 'P3', 'P4'])

    # Combining descriptors calculated by Multiwfn and morfeus
    df_total = pd.concat([df_Multiwfn, df_morfeus], axis=1)

    print(df_total)
    df_total.to_csv('Physical_organic_descriptors.csv', index=False)

    print('\nInfo: The results were saved in the following path:\n      {0}'.format(os.getcwd() + os.sep + 'Physical_organic_descriptors.csv'))


def total_running_time(end_time, start_time):
    tot_seconds = round(end_time - start_time, 2)
    days = tot_seconds // 86400
    hours = (tot_seconds % 86400) // 3600
    minutes = (tot_seconds % 86400 % 3600)// 60
    seconds = tot_seconds % 60
    print(">> Elapsed time: {0:2d} day(s) {1:2d} hour(s) {2:2d} minute(s) {3:5.2f} second(s) <<".format(int(days),int(hours),int(minutes),seconds))

def descriptor_info():
    print('\n=============================   Monomer Radical Descriptor Information   ======================================')
    print('> 1. Frontier orbital energy:')
    print('    + E1 : SOMO energy of monomer radical')
    print('    + E2 : The first ionization energy of monomer radical')
    print('    + E3 : Electron affinity of monomer radical')
    print('    + E4 : Electronegativity of monomer radical')
    print('    + E5: Chemical softness of monomer radical')
    print('\n> 2. Bond order:')
    print('    + B1 : Maximum Wiberg bond index of carbon radical atom')
    print('    + B2 : Minimum Wiberg bond index of carbon radical atom')
    print('    + B3 : Average Wiberg bond index of carbon radical atom')
    print('\n> 3. Atomic charge:')
    print('    + Q1 : Atomic charge of carbon radical atom')
    print('    + Q2 : The condensed-to-atom Fukui functions (𝑓c0) of carbon radical atom')
    print('    + Q3 : The condensed-to-atom softness (sigma c) of carbon radical atom')
    print('\n> 4. Buried volume:')
    print('    + V1 : Buried volumes of carbon radical atom, sphere radius = 3 angstrom')
    print('    + V2 : Buried volumes of carbon radical atom, sphere radius = 4 angstrom')
    print('    + V3 : Buried volumes of carbon radical atom, sphere radius = 5 angstrom')
    print('\n> 5. Dispersion:')
    print('    + P1 : Average molecular dispersion potential, Pint of the monomer radical')
    print('    + P2 : Maximum dispersion parameter P on the molecular surface of the monomer radical')
    print('    + P3 : Minimum dispersion parameter P on the molecular surface of the monomer radical')
    print('    + P4 : The standard deviation of dispersion parameter P on the molecular surface of the monomer radical')
    print('================================================================================================================')


if __name__ == '__main__':
    start_time = time.time()
    start_date = datetime.datetime.now()
    print('      ***  The \'PhysOrg-monomer\' script (Linux) started at {0}  ***\n'.format(start_date.strftime("%Y-%m-%d %H:%M:%S")))

    try:
        current_path = os.getcwd()
        target_file_name = r".*\.fchk$"
        all_target_files = search(current_path, target_file_name)
        result_file_list_Multiwfn = call_Multiwfn(all_target_files)
        result_file_list_morfeus = call_morfeus(all_target_files)
        data_extraction(result_file_list_Multiwfn, result_file_list_morfeus)
        descriptor_info()
    except Exception as e:
        print(f'Error: {e}')
        print('Note that: (1) The "isilent= 1" MUST be set in the settings.ini of Multiwfn.')

    end_time = time.time()
    end_date = datetime.datetime.now()
    print('\n      ***  The \'PhysOrg-monomer\' script (Linux) terminated at {0}  ***\n'.format(end_date.strftime("%Y-%m-%d %H:%M:%S")))
    total_running_time(end_time, start_time)
        