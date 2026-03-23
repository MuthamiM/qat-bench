import argparse
import subprocess
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))

def run_script(script_path):
    print(f"\n[{script_path}] -> Executing module...")
    env = os.environ.copy()
    env["PYTHONPATH"] = base_dir
    subprocess.run([sys.executable, script_path], check=True, cwd=base_dir, env=env)

def main():
    parser = argparse.ArgumentParser(description="QAT-Bench Pipeline Orchestrator")
    parser.add_argument("--mode", choices=['train', 'qat', 'ptq', 'bench', 'report', 'all'], required=True, 
                        help="Select the module to execute.")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(__file__)
    
    if args.mode in ['train', 'all']:
        run_script(os.path.join(base_dir, 'data', 'prepare.py'))
        run_script(os.path.join(base_dir, 'train', 'trainer.py'))
        
    if args.mode in ['qat', 'all']:
        run_script(os.path.join(base_dir, 'train', 'qat_trainer.py'))
        
    if args.mode in ['ptq', 'all']:
        run_script(os.path.join(base_dir, 'bench', 'ptq.py'))
        
    if args.mode in ['bench', 'all']:
        run_script(os.path.join(base_dir, 'bench', 'evaluator.py'))
        
    if args.mode in ['report', 'all']:
        run_script(os.path.join(base_dir, 'report', 'generator.py'))
        run_script(os.path.join(base_dir, 'report', 'visualizer.py'))

if __name__ == "__main__":
    main()
