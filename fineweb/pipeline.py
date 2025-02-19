from datasets import load_dataset
import pandas as pd

# Load the dataset in streaming mode
dataset = load_dataset("HuggingFaceFW/fineweb", 
                      name="sample-10BT", 
                      split="train", 
                      streaming=True)

# Process in batches and write simultaneously
chunk_size = 10000

# Open file in append mode
with open('train.txt', 'w') as f:  # 'w' to create new file initially
    for i, batch in enumerate(dataset.iter(batch_size=chunk_size)):
        # Convert batch to DataFrame and save immediately
        df = pd.DataFrame(batch['text'])
        # Append to file without index and header
        df.to_csv(f, mode='a', index=False, header=False, sep='\t')
        
        if i % 10 == 0:
            print(f"Processed {(i+1)*chunk_size} examples")