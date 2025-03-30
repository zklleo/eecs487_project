import requests
import os

# Create directory for dataset
os.makedirs('data/hotpotqa', exist_ok=True)

# URLs for the dataset files
urls = {
    'train': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json',
    'dev': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
    'test': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json'
}

# Download each file
for split, url in urls.items():
    print(f"Downloading {split} set...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        
        with open(f'data/hotpotqa/hotpot_{split}.json', 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {split} set successfully!")
        
        # Print file size
        file_size = os.path.getsize(f'data/hotpotqa/hotpot_{split}.json') / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Error downloading {split} set: {e}")