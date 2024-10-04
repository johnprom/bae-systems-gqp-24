from pipeline import run_pipeline
from util.util import load_pipeline_config
 
def main():
    config = load_pipeline_config()
    
    if "execution" in config:
        run_pipeline(config)
    else:
        print("Must set execution list in configuration file")
    
    print("Exited Main.")

if __name__ == "__main__":
    main()
