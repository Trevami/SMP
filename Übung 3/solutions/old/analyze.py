#!/usr/bin/env python3

import pickle
import argparse
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file', help="Path to pickle file.")
args = parser.parse_args()

with open(args.file, 'rb') as fp:
    data = pickle.load(fp)


def running_average(O, M) -> np.ndarray:
    O_npa = np.array(O.copy())
    avg_time_series = []
    for i in range(M, len(O_npa) - M):
        avg_time_series.append(np.average(O_npa[i - M: i + M]))
    return np.array(avg_time_series)


def remaining_avg(teq):
    DT = data['delta_time']
    index = int(np.ceil(teq/DT)) if teq > 0 else 0
    avg_energy = np.average(np.array(data['energies'])[index:])
    avg_pressure = np.average(np.array(data['pressures'])[index:])
    avg_temperature = np.average(np.array(data['temperatures'])[index:])

    return avg_energy, avg_pressure, avg_temperature


def check_rel_convergence(output_path, check_timestep, E_threshold=1e-3, P_threshold=1e-3,
                          T_threshold=1e-3, all_thresholds=None, extra_it=5,):
    teq = 0.0
    DT = data['delta_time']

    # Determine the check range parameters
    max_timestep = len(data['energies']) * DT
    num_steps = int(np.floor(max_timestep/check_timestep))
    # Raise value error if the check timestep is larger than the simulated regeion
    if num_steps < 1:
        raise ValueError(
            "Check timestep is lager then the simmulated widow. Choose a smaller one!")
    if check_timestep < DT:
        raise ValueError(
            "Check timestep is smaller then the itegration window. Choose a larger one!")

    # Initializaion
    if all_thresholds:
        thresholds = [all_thresholds for _ in range(3)]
    else:
        thresholds = [E_threshold, P_threshold, T_threshold]
    val_list0 = list(remaining_avg(teq))
    teq += check_timestep
    converged_teq = None
    converged_avgs = None

    with open(output_path, "w") as file:

        # Check for convergence iterativly
        file.write("{:^47}".format("Convergence Check")+"\n")
        file.write("{:<5}    {:^11}   {:^11}   {:^11}".format(
            "IT", "Rel. DE", "Rel. DP", "Rel. DT") + "\n")
        file.write(("-" * 47) + "\n")

        for i in range(num_steps - 1):
            val_list = list(remaining_avg(teq))
            conv_list = []

            file.write("{:<5}".format(i + 1))
            for val0, val, threshold in zip(val_list0, val_list, thresholds):
                conv_val = abs(val - val0) / abs(val0)

                # Check for convergence
                file.write("{:>14.4E}".format(conv_val))
                if conv_val <= threshold:
                    conv_list.append(True)
            file.write("\n")

            if len(conv_list) == 3:
                converged_teq = teq
                converged_avgs = val_list
                # Buffer if values are stable only in a small region
                if extra_it == 0:
                    break
                extra_it -= 1
            else:
                converged_teq = None

            val_list0 = val_list
            teq += check_timestep

        file.write(("-" * 47) + "\n")
        if converged_teq:
            file.write("{:^47}".format(
                f"Converged at the time %.2f a.u." % teq) + "\n")
            file.write(("-" * 47) + "\n")
            file.write("Warmup stage ended according to set thresholds:" + "\n")
            file.write("{:^16}{:^15}{:^16}".format(
                "Rel. DE", "Rel. DP", "Rel. DT") + "\n")
            file.write("{:^16.1E}{:^15.1E}{:^16.1E}".format(
                thresholds[0], thresholds[1], thresholds[2]) + "\n")
            file.write("-" * 47 + "\n")
            file.write("Averaged values after this time:" + "\n")
            file.write("System energy      E_tot :{:>21.4f}".format(
                converged_avgs[0]) + "\n")
            file.write("System pressure    P     :{:>21.4f}".format(
                converged_avgs[1]) + "\n")
            file.write("System temperature T     :{:>21.4E}".format(
                converged_avgs[2]) + "\n")
            return converged_teq
        else:
            file.write("{:^47}".format(f" Did not converge!") + "\n")
            return None


def plot_data(plot_name, M=None, warmup=None):
    DT = data['delta_time']
    energies = data['energies']
    pressures = data['pressures']
    temperatures = data['temperatures']

    min_E, max_E = np.min(energies), np.max(energies)
    delta_E = abs(max_E - min_E)
    min_P, max_P = np.min(pressures), np.max(pressures)
    delta_P = abs(max_P - min_P)
    min_T, max_T = np.min(temperatures), np.max(temperatures)
    delta_T = abs(max_T - min_T)

    time_range_E = np.arange(len(energies))*DT
    min_x_E, max_x_E = 0.0, len(energies)*DT
    time_range_P = np.arange(len(pressures))*DT
    min_x_P, max_x_P = 0.0, len(pressures)*DT
    time_range_T = np.arange(len(temperatures))*DT
    min_x_T, max_x_T = 0.0, len(temperatures)*DT

    x_label = "time in a.u."
    y_label = ""

    if M:
        energies = running_average(energies, M)
        pressures = running_average(pressures, M)
        temperatures = running_average(temperatures, M)

        # Adjustign the plot ranges for averaged values
        time_range_E = time_range_E[M:-M]
        time_range_P = time_range_P[M:-M]
        time_range_T = time_range_T[M:-M]
        y_label = f"(avg. M={M}) "

    plot_path = Path(__file__).resolve().parent.parent/'plots'
    time_tick_spacing = (0.4)

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(2, 2)
    b_offst = 0.1

    # -------- Plot for Energy --------
    ax_1 = fig.add_subplot(grid[0, :])
    ax_1.plot(time_range_E, energies)
    ax_1.set_ylabel(y_label + "energy in a.u.")
    ax_1.set_title("Energy")
    ax_1.set_xlim(min_x_E, max_x_E)
    ax_1.set_ylim(min_E - delta_E * b_offst, max_E + delta_E * b_offst)

    # ------- Plot for Pressure -------
    ax_2 = fig.add_subplot(grid[1, 0])
    ax_2.plot(time_range_P, pressures)
    ax_2.set_ylabel(y_label + "pressure in a.u.")
    ax_2.set_title("Pressure")
    ax_2.set_xlim(min_x_P, max_x_P)
    ax_2.set_ylim(min_P - delta_P * b_offst, max_P + delta_P * b_offst)

    # ------ Plot for Temperature -----
    ax_3 = fig.add_subplot(grid[1, 1])
    ax_3.plot(time_range_T, temperatures)
    ax_3.set_ylabel(y_label + "temperature in a.u.")
    ax_3.set_title("Temperature")
    ax_3.set_xlim(min_x_T, max_x_T)
    ax_3.set_ylim(min_T - delta_T * b_offst, max_T + delta_T * b_offst)

    axis = [ax_1, ax_2, ax_3]

    for ax in axis:
        ax.set_xlabel(x_label)

    if warmup:
        avg_energy, avg_pressure, avg_temperature = remaining_avg(warmup)
        for ax in axis:
            ax.axvline(x=warmup, color='gray', linestyle='--')
            ax.axvspan(xmin=0, xmax=warmup, facecolor='gray',
                       alpha=0.15, label="Warmup of the system")
    else:
        avg_energy, avg_pressure, avg_temperature = remaining_avg(0.0)

    ax_1.axhline(y=avg_energy, color='red', alpha=0.3, linestyle='-')
    ax_2.axhline(y=avg_pressure, color='red', alpha=0.3, linestyle='-')
    ax_3.axhline(y=avg_temperature, color='red', alpha=0.3, linestyle='-')

    fig.set_dpi(100)
    plt.tight_layout()
    plt.savefig(plot_path/plot_name)

if __name__ == '__main__':
    from pathlib import Path

    # conv_output_path = Path(__file__).resolve(
    # ).parent.parent/'warmup_convergence_logs'/f"{args.file}.txt"
    # x_warmup = check_rel_convergence(
    #     conv_output_path,
    #     data['delta_time'],
    #     E_threshold=1.5e-5,
    #     P_threshold=5e-3,
    #     T_threshold=1e-4,
    #     extra_it=10)

    # plot_data(f"{args.file}_plot_1.png", warmup=x_warmup)
    # plot_data(f"{args.file}_plot_2.png", 10, warmup=x_warmup)
    # plot_data(f"{args.file}_plot_3.png", 100, warmup=x_warmup)

    rdfs = np.array(data['rdfs'])
    avg_rdfs = np.mean(rdfs[:, :, :], axis=0)
    plt.plot(avg_rdfs[0, :], avg_rdfs[1, :])
    plt.show()