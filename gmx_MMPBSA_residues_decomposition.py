#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
from io import StringIO
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging and logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gmx_MMPBSA_residueal_decomposition.log.txt')
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting the script...")
logger.info("Python version: %s", sys.version)
logger.info("Numpy version: %s", np.__version__)
logger.info("Pandas version: %s", pd.__version__)

# Read the dat file
def parse_decomp_energy_file(file_path):
    """Parse the decomposed energy file and return a DataFrame."""
    with open(file_path, 'r') as f:
        decomp_data = f.read()
    f.close()
    
    # Use StringIO to simulate reading from a file
    data_io = StringIO(decomp_data)

    # Set column names
    column_names = [
        'Residue', 'Internal_Avg', 'Internal_Std_Dev', 'Internal_Std_Err', 
        'Van_der_Waals_Avg', 'Van_der_Waals_Std_Dev', 'Van_der_Waals_Std_Err', 
        'Electrostatic_Avg', 'Electrostatic_Std_Dev', 'Electrostatic_Std_Err', 
        'Polar_Solv_Avg', 'Polar_Solv_Std_Dev', 'Polar_Solv_Std_Err', 
        'Non_Polar_Solv_Avg', 'Non_Polar_Solv_Std_Dev', 'Non_Polar_Solv_Std_Err', 
        'Total_Avg', 'Total_Std_Dev', 'Total_Std_Err'
    ]
    # Skip the first 6 lines of metadata and set the column headers
    decomp_df = pd.read_csv(data_io, skiprows=3, names=column_names)
    
    return decomp_df

def parse_decomp_energy_terms(decomp_df, complex_energy=False, receptor_energy=False, ligand_energy=False, delta_energy=False):
    """
    Parse decomposed energy terms from the DataFrame.

    Args:
        decomp_df (DataFrame): DataFrame containing the decomposed energy data.
        complex_energy (bool): If True, return complex energy DataFrames.
        receptor_energy (bool): If True, return receptor energy DataFrames.
        ligand_energy (bool): If True, return ligand energy DataFrames.
        delta_energy (bool): If True, return delta energy DataFrames.

    Returns:
        tuple: Selected DataFrames based on the requested energy type(s).
    """
    
    # Load decomposed energy file
    decomp_df = parse_decomp_energy_file(decomp_df)

    # Check if the DataFrame is empty
    if decomp_df.empty:
        print("The DataFrame is empty. Please check the input file.")
        return None, None, None

    # --- Complex section ---
    poisson_boltzmann_model_index = decomp_df[decomp_df['Residue'] == 'Energy Decomposition Analysis (All units kcal/mol): Poisson Boltzmann model'].index[0]
    poisson_boltzmann_model_df = decomp_df.iloc[poisson_boltzmann_model_index + 1:]

    mpb_cmplx_total_index = poisson_boltzmann_model_df[poisson_boltzmann_model_df['Residue'] == 'Total Energy Decomposition:'].index[0]
    mpb_cmplx_sidechain_index = poisson_boltzmann_model_df[poisson_boltzmann_model_df['Residue'] == 'Sidechain Energy Decomposition:'].index[0]
    mpb_complx_backbone_index = poisson_boltzmann_model_df[poisson_boltzmann_model_df['Residue'] == 'Backbone Energy Decomposition:'].index[0]
    mpb_receptor_index = poisson_boltzmann_model_df[poisson_boltzmann_model_df['Residue'] == 'Receptor:'].index[0]

    complex_total_df = decomp_df.iloc[mpb_cmplx_total_index + 3: mpb_cmplx_sidechain_index]
    complex_sidechain_df = decomp_df.iloc[mpb_cmplx_sidechain_index + 3: mpb_complx_backbone_index]
    complex_backbone_df = decomp_df.iloc[mpb_complx_backbone_index + 3: mpb_receptor_index]

    # --- Receptor section ---
    poisson_boltzmann_receptor_df = decomp_df.iloc[mpb_receptor_index + 1:]
    mpb_receptor_total_index = poisson_boltzmann_receptor_df[poisson_boltzmann_receptor_df['Residue'] == 'Total Energy Decomposition:'].index[0]
    mpb_receptor_sidechain_index = poisson_boltzmann_receptor_df[poisson_boltzmann_receptor_df['Residue'] == 'Sidechain Energy Decomposition:'].index[0]
    mpb_receptor_backbone_index = poisson_boltzmann_receptor_df[poisson_boltzmann_receptor_df['Residue'] == 'Backbone Energy Decomposition:'].index[0]
    mpb_ligand_index = poisson_boltzmann_receptor_df[poisson_boltzmann_receptor_df['Residue'] == 'Ligand:'].index[0]

    receptor_total_df = decomp_df.iloc[mpb_receptor_total_index + 3: mpb_receptor_sidechain_index]
    receptor_sidechain_df = decomp_df.iloc[mpb_receptor_sidechain_index + 3: mpb_receptor_backbone_index]
    receptor_backbone_df = decomp_df.iloc[mpb_receptor_backbone_index + 3: mpb_ligand_index]

    # --- Ligand section ---
    poisson_boltzmann_ligand_df = decomp_df.iloc[mpb_ligand_index + 1:]
    mpb_ligand_total_index = poisson_boltzmann_ligand_df[poisson_boltzmann_ligand_df['Residue'] == 'Total Energy Decomposition:'].index[0]
    mpb_ligand_sidechain_index = poisson_boltzmann_ligand_df[poisson_boltzmann_ligand_df['Residue'] == 'Sidechain Energy Decomposition:'].index[0]
    mpb_ligand_backbone_index = poisson_boltzmann_ligand_df[poisson_boltzmann_ligand_df['Residue'] == 'Backbone Energy Decomposition:'].index[0]
    mpb_deltas_index = poisson_boltzmann_ligand_df[poisson_boltzmann_ligand_df['Residue'] == 'DELTAS:'].index[0]

    ligand_total_df = decomp_df.iloc[mpb_ligand_total_index + 3: mpb_ligand_sidechain_index]
    ligand_sidechain_df = decomp_df.iloc[mpb_ligand_sidechain_index + 3: mpb_ligand_backbone_index]
    ligand_backbone_df = decomp_df.iloc[mpb_ligand_backbone_index + 3: mpb_deltas_index]

    # --- Delta section ---
    poisson_boltzmann_delta_df = decomp_df.iloc[mpb_deltas_index + 1:]
    mpb_delta_total_index = poisson_boltzmann_delta_df[poisson_boltzmann_delta_df['Residue'] == 'Total Energy Decomposition:'].index[0]
    mpb_delta_sidechain_index = poisson_boltzmann_delta_df[poisson_boltzmann_delta_df['Residue'] == 'Sidechain Energy Decomposition:'].index[0]
    mpb_delta_backbone_index = poisson_boltzmann_delta_df[poisson_boltzmann_delta_df['Residue'] == 'Backbone Energy Decomposition:'].index[0]

    delta_total_df = decomp_df.iloc[mpb_delta_total_index + 3: mpb_delta_sidechain_index]
    delta_sidechain_df = decomp_df.iloc[mpb_delta_sidechain_index + 3: mpb_delta_backbone_index]
    delta_backbone_df = decomp_df.iloc[mpb_delta_backbone_index + 3:]


    # --- Return logic ---
    if complex_energy:
        # --- Logging ---
        logging.info(f"MPB COMPLEX TOTAL ENERGY:{mpb_cmplx_total_index}")
        logging.info(f"MPB COMPLEX SIDECHAIN ENERGY:{mpb_cmplx_sidechain_index}")
        logging.info(f"MPB COMPLEX BACKBONE ENERGY:{mpb_complx_backbone_index}")
        return complex_total_df, complex_backbone_df, complex_sidechain_df
    elif receptor_energy:
        # --- Logging ---
        logging.info(f"MPB RECEPTOR TOTAL ENERGY:{mpb_receptor_total_index}")
        logging.info(f"MPB RECEPTOR SIDECHAIN ENERGY:{mpb_receptor_sidechain_index}")
        logging.info(f"MPB RECEPTOR BACKBONE ENERGY:{mpb_receptor_backbone_index}")
        return receptor_total_df, receptor_backbone_df, receptor_sidechain_df
    elif ligand_energy:
        # --- Logging ---
        logging.info(f"MPB LIGAND TOTAL ENERGY:{mpb_ligand_total_index}")
        logging.info(f"MPB LIGAND SIDECHAIN ENERGY:{mpb_ligand_sidechain_index}")
        logging.info(f"MPB LIGAND BACKBONE ENERGY:{mpb_ligand_backbone_index}")
        return ligand_total_df, ligand_backbone_df, ligand_sidechain_df
    elif delta_energy:
        # --- Logging ---
        logging.info(f"MPB DELTA TOTAL ENERGY:{mpb_delta_total_index}")
        logging.info(f"MPB DELTA SIDECHAIN ENERGY:{mpb_delta_sidechain_index}")
        logging.info(f"MPB DELTA BACKBONE ENERGY:{mpb_delta_backbone_index}")
        return delta_total_df, delta_backbone_df, delta_sidechain_df
    else:
        print("❗ Please specify one energy type to return (complex_energy, receptor_energy, ligand_energy, or delta_energy).")
        return None, None, None

def plot_protein_delta_decomposition_energy(df, output_file='delta_decomposition_energy.png'):
    """
    Plots the average total energy decomposition per amino acid from a given dataframe.

    Parameters:
    - df: pandas.DataFrame
        DataFrame containing the columns:
        ['Residue', 'Internal_Avg', 'Van_der_Waals_Avg', 'Electrostatic_Avg', 
         'Polar_Solv_Avg', 'Non_Polar_Solv_Avg', 'Total_Avg']
    - output_file: str
        Path to save the generated plot image.

    Returns:
    - None
    """

    # Columns to use
    plot_columns = ['Residue', 'Internal_Avg', 'Van_der_Waals_Avg', 'Electrostatic_Avg', 
                    'Polar_Solv_Avg', 'Non_Polar_Solv_Avg', 'Total_Avg']

    # Filter residues starting with 'R'
    get_data_protein = df[df['Residue'].str.startswith("R")]

    # Select relevant columns
    plot_df = pd.DataFrame(get_data_protein, columns=plot_columns)

    # Extract amino acid names
    plot_df['Amino_Acid'] = plot_df['Residue'].str[3:10]

    # Ensure numeric columns
    for col in plot_columns[1:]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce', downcast='float')

    # Prepare for plotting
    updated_plot_df = plot_df[['Amino_Acid', 'Internal_Avg', 'Van_der_Waals_Avg', 
                               'Electrostatic_Avg', 'Polar_Solv_Avg', 
                               'Non_Polar_Solv_Avg', 'Total_Avg']]
    updated_plot_df.set_index('Amino_Acid', inplace=True)

    # --- Visualization ---
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    bars = updated_plot_df['Total_Avg'].sort_values(ascending=False).plot(
        kind='bar', 
        color=sns.color_palette("pastel")[2], 
        edgecolor='black', 
        ax=ax,
        width=0.6
    )

    # Improve labels and title
    #ax.set_title('Δ Decomposition Energy of Amino Acids', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Amino Acid', fontsize=14)
    ax.set_ylabel('Average Δ Energy (kcal/mol)', fontsize=14)
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    # Annotate values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            # For positive bars: annotate just above; for negative: just below
            offset = 3  # points
            va = 'bottom' if height >= 0 else 'top'
            ax.annotate(
                f'{height:.1f}',
                (p.get_x() + p.get_width() / 2, height),
                xytext=(0, offset if height >= 0 else -offset),
                textcoords='offset points',
                ha='center',
                va=va,
                fontsize=10,
                rotation=0  # or small angle if crowded
            )

    # Save the plot
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved as :{output_file}")
    plt.close()

def plot_rna_delta_decomposition_energy(df, output_file='rna_delta_decomposition_energy.png'):
    """
    Plots the average total energy decomposition per RNA base from a given dataframe.

    Parameters:
    - df: pandas.DataFrame
        DataFrame containing the columns:
        ['Residue', 'Internal_Avg', 'Van_der_Waals_Avg', 'Electrostatic_Avg', 
         'Polar_Solv_Avg', 'Non_Polar_Solv_Avg', 'Total_Avg']
    - output_file: str
        Path to save the generated plot image.

    Returns:
    - None
    """

    # Define relevant columns
    plot_columns = ['Residue', 'Internal_Avg', 'Van_der_Waals_Avg', 'Electrostatic_Avg', 
                    'Polar_Solv_Avg', 'Non_Polar_Solv_Avg', 'Total_Avg']

    # Filter residues starting with 'L' (for RNA)
    get_data_rna = df[df['Residue'].str.startswith("L")]

    # Select needed columns
    plot_df = pd.DataFrame(get_data_rna, columns=plot_columns)

    # Extract base names
    plot_df['Base'] = plot_df['Residue'].str[3:10]

    # Ensure numeric columns
    for col in plot_columns[1:]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce', downcast='float')

    # Organize for plotting
    updated_plot_df = plot_df[['Base', 'Internal_Avg', 'Van_der_Waals_Avg', 
                               'Electrostatic_Avg', 'Polar_Solv_Avg', 
                               'Non_Polar_Solv_Avg', 'Total_Avg']]
    updated_plot_df.set_index('Base', inplace=True)

    # --- Visualization ---
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = updated_plot_df['Total_Avg'].sort_values(ascending=False).plot(
        kind='bar', 
        color=sns.color_palette("pastel")[0], 
        edgecolor='black', 
        ax=ax,
        width=0.6
    )

    # Improve labels and title
    #ax.set_title('Δ Decomposition Energy of RNA Nucleotides', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Nucleotide', fontsize=14)
    ax.set_ylabel('Average Δ Energy (kcal/mol)', fontsize=14)
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

    # Annotate values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            # For positive bars: annotate just above; for negative: just below
            offset = 3  # points
            va = 'bottom' if height >= 0 else 'top'
            ax.annotate(
                f'{height:.1f}',
                (p.get_x() + p.get_width() / 2, height),
                xytext=(0, offset if height >= 0 else -offset),
                textcoords='offset points',
                ha='center',
                va=va,
                fontsize=10,
                rotation=0  # or small angle if crowded
            )

    # Remove default legend if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"RNA plot saved as :{output_file}")
    plt.close()

def main():
    # DECOMPOSED ENERGY FILE PATHS
    decomp_wt_fh = "/home/shrikant/my_work/OBJ4/SF3A1_UBL/Binding-Energy-Decomposition/wt/8ID2_WT_DECOMP_MMPBSA.dat"
    decomp_r788_df = "/home/shrikant/my_work/OBJ4/SF3A1_UBL/Binding-Energy-Decomposition/788A/8ID2_R788A_DECOMP_MMPBSA.dat"
    decomp_r791_df = "/home/shrikant/my_work/OBJ4/SF3A1_UBL/Binding-Energy-Decomposition/791A/8ID2_R791A_DECOMP_MMPBSA.dat"
    decomp_r788_791_df = "/home/shrikant/my_work/OBJ4/SF3A1_UBL/Binding-Energy-Decomposition/788A-791A/8ID2_R788A_R791A_DECOMP_MMPBSA.dat"

    # Parse the decomposed energy terms
    wt_delta_total, wt_delta_backbone, wt_delta_sidechain = parse_decomp_energy_terms(decomp_wt_fh, delta_energy=True)
    r788a_delta_total, r788a_delta_backbone, r788a_delta_sidechain = parse_decomp_energy_terms(decomp_r788_df, delta_energy=True)
    r791a_delta_total, r791a_delta_backbone, r791a_delta_sidechain = parse_decomp_energy_terms(decomp_r791_df, delta_energy=True)
    r788_791a_delta_total, r788_791a_delta_backbone, r788_791a_delta_sidechain = parse_decomp_energy_terms(decomp_r788_791_df, delta_energy=True)

    # Prepare a dictionary of all mutation datasets
    mutation_data = {
        "WT": [wt_delta_total, wt_delta_backbone, wt_delta_sidechain],
        "R788A": [r788a_delta_total, r788a_delta_backbone, r788a_delta_sidechain],
        "R791A": [r791a_delta_total, r791a_delta_backbone, r791a_delta_sidechain],
        "R788A_R791A": [r788_791a_delta_total, r788_791a_delta_backbone, r788_791a_delta_sidechain]
    }

    # Labels for the energy parts
    labels = ["total", "backbone", "sidechain"]

    # Loop through each mutation and each energy type
    for mutation_name, energy_dfs in mutation_data.items():
        for label, df in zip(labels, energy_dfs):
            output_filename = f"delta_decomp_energy_{mutation_name}_{label}.png"
            plot_protein_delta_decomposition_energy(df, output_file=output_filename)
    
    # Loop through each mutation and each energy type
    for mutation_name, energy_dfs in mutation_data.items():
        for label, df in zip(labels, energy_dfs):
            output_filename = f"delta_decomp_energy_rna_{mutation_name}_{label}.png"
            plot_rna_delta_decomposition_energy(df, output_file=output_filename)

    # --- End of main function ---
    logger.info("Script completed successfully.")
    logger.info("All plots saved successfully.")
    logger.info("Exiting the script.")
    plt.close('all')

if __name__ == "__main__":
    main()
    # --- End of script ---
    logger.info("Script execution finished.")

