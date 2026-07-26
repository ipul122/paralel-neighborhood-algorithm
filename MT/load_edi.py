import os
import sys

import numpy as np
np.float = float
np.int = int
np.complex = complex

from mtpy.core.mt import MT
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("\n[ERROR] Format salah! Seret file .edi Anda ke terminal.")
    print("Contoh penggunaan: python load_edi.py <drag_file_ke_sini>\n")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"\n[ERROR] File tidak ditemukan di lokasi: {file_path}\n")
    sys.exit(1)

try:
    mt = MT(file_path)

    freq = mt.Z.freq
    res_eff = mt.Z.res_det
    phase_eff = mt.Z.phase_det
    periode = 1.0 / freq

    name_file = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n>>> Sukses memuat data: {name_file}.edi")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input = input("untuk NA atau LM? (ketik 'NA' atau 'LM'): ").strip().upper()

    if input == "NA":
        output_filename = "CUSTOM_NA.txt"
    elif input == "LM":
        output_filename = "CUSTOM_LM.txt"
    else:
        print("\n[ERROR] Input tidak valid! Harap ketik 'NA' atau 'LM'.\n")
        sys.exit(1)

    output_path = os.path.join(current_dir, output_filename)

    export_data = np.column_stack((freq, res_eff, phase_eff))

    header_text = f"Frequency(Hz_{name_file})\tAppRes(Ohm.m)\tPhase(deg)"

    np.savetxt(
        output_path, 
        export_data, 
        delimiter="\t", 
        header=header_text, 
        comments=""  
    )

    print("="*55)
    print(f"[BERHASIL] Data diekstraksi ke: {output_filename}")
    print(f"Header kolom  : {header_text}")
    print(f"Lokasi lengkap: {output_path}")
    print("="*55 + "\n")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.loglog(periode, res_eff, 'ro-', label='AppRes (Zdet)')
    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("Apparent Resistivity (Ωm)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.semilogx(periode, phase_eff, 'bo-', label='Phase (Zdet)')
    ax2.set_ylabel("Phase (deg)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f"Invariant - {name_file}")
    plt.show()

        
    freq = mt.Z.freq
    period = 1/freq

    res_te = mt.Z.res_xy
    res_tm = mt.Z.res_yx
    res_inv = mt.Z.res_det

    phase_te = mt.Z.phase_xy
    phase_tm = mt.Z.phase_yx
    phase_inv = mt.Z.phase_det

    phase_tm_corr = phase_tm + 180

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.loglog(period, res_te, 'b^-', label='Res TE (xy)')
    ax1.loglog(period, res_tm, 'bs-', label='Res TM (yx)')
    ax1.loglog(period, res_inv, 'ro-', label='Res Invariant', linewidth=2)

    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("Apparent Resistivity (Ωm)")
    ax1.set_title(f"Apparent Resistivity")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.semilogx(period, phase_te, 'b^-', label='Phase TE (xy)')
    ax2.semilogx(period, phase_tm_corr, 'bs-', label='Phase TM (yx)')
    ax2.semilogx(period, phase_inv, 'ro-', label='Phase Invariant', linewidth=2)

    ax2.set_xlabel("Period (s)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title(f"Phase")
    ax2.set_ylim(0, 100)
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    plt.show()

except Exception as e:
    print(f"\n[ERROR] Gagal memproses data MT karena: {e}\n")