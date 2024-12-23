from src.preprocess import main as preprocess
from src.train import main as train
from src.evaluate import main as evaluate
from src.visualise import main as visualize

def main():
    print("Step 1: Preprocessing the data...")
    # preprocess()
    
    print("Step 2: Training models...")
    train()
    
    print("Step 3: Evaluating models...")
    metrics, confusion_matrices = evaluate()
    
    print("Step 4: Visualizing results...")
    visualize()

if __name__ == "__main__":
    main()
