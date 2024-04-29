import argparse
import os
from typing import Set, Tuple


def find_unique_files(dataset_path: str) -> Tuple[Set[str], Set[str], int, int]:
    """
    Trova i file unici nelle cartelle "clean" e "noisy".

    Parametri:
    dataset_path (str): Il percorso al dataset. Deve contenere le sottocartelle "clean" e "noisy".

    Restituisce:
    Tuple[Set[str], Set[str]]: Due insiemi di nomi di file. Il primo insieme contiene i nomi dei file unici
    nella cartella "clean". Il secondo insieme contiene i nomi dei file unici nella cartella "noisy".
    """
    # Percorso alle cartelle clean e noisy
    clean_dir = os.path.join(dataset_path, "clean_new")
    noisy_dir = os.path.join(dataset_path, "noisy_new")

    # Ottieni i nomi dei file .png in ciascuna directory
    clean_files = set(f for f in os.listdir(clean_dir) if f.endswith(".png"))
    noisy_files = set(f for f in os.listdir(noisy_dir) if f.endswith(".png"))
    
    # Trova i file unici in ciascuna directory
    unique_clean = clean_files - noisy_files
    unique_noisy = noisy_files - clean_files

    n_clean = len(clean_files)
    n_noisy = len(noisy_files)

    return unique_clean, unique_noisy, n_clean, n_noisy


def main():
    parser = argparse.ArgumentParser(
        description="Trova file unici nelle cartelle clean e noisy"
    )
    parser.add_argument("dataset_path", type=str, help="Il percorso al dataset")

    args = parser.parse_args()

    unique_clean, unique_noisy, n_clean, n_noisy = find_unique_files(args.dataset_path)

    print(
        f"Found {len(unique_clean)}/{n_clean} files unique to clean directory:\n",
        unique_clean,
    )
    print(
        f"Found {len(unique_noisy)}/{n_noisy} files unique to noisy directory:\n",
        unique_noisy,
    )


if __name__ == "__main__":
    main()
