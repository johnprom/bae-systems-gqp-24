from pipeline import run_pipeline
from util.util import load_pipeline_config
 
def main():
    config = load_pipeline_config()
    run_pipeline(config)
    
    print("Exited Main.")

if __name__ == "__main__":
    main()
