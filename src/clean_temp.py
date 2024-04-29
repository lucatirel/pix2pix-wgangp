import os
import argparse
import shutil

def is_checkpoints_empty(run_path):
    try:
        checkpoints_path = os.path.join(run_path, 'checkpoints')
        return not os.listdir(checkpoints_path)
    except:
        return True

def count_and_print_empty_checkpoints(root_path, delete_flag=False):
    count_empty = 0
    for run in os.listdir(root_path):
        run_path = os.path.join(root_path, run)
        if os.path.isdir(run_path):
            if is_checkpoints_empty(run_path):
                count_empty += 1
                print(run_path)
                if delete_flag:
                    shutil.rmtree(os.path.join(run_path))

    print(f"Totale cartelle 'checkpoints' vuote: {count_empty}/{len(os.listdir(root_path))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to find and optionally delete empty 'checkpoints' directories.")
    parser.add_argument("root_directory", type=str, help="Path to the root directory of the runs.")
    parser.add_argument("--delete", action="store_true", help="Flag to delete the empty 'checkpoints' directories.")
    args = parser.parse_args()

    count_and_print_empty_checkpoints(args.root_directory, delete_flag=args.delete)


# python clean_temp.py C:\Users\luca9\Desktop\Pix2pix WGAN-GP\Code\Runs_old\Runs --delete
