from pipeline import run_pipeline
from util.util import load_config

def main():
    config = load_config('../pipeline_config.yaml')
    run_pipeline(config)
    
    print("Exited Main.")

if __name__ == "__main__":
    main()
