import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config

def download_sn8_sample(target_dir='data/SN8'):
    """
    Downloads a sample of SpaceNet-8 data from S3.
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'spacenet-dataset'
    prefix = 'spacenet/SN8_floods/'
    
    # List objects
    print(f"Listing objects in {bucket_name}/{prefix}...")
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        count = 0
        limit = 20 # Limit to a few files for a 'sample'
        
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('/'):
                    continue
                
                # We want typical training data structure
                # Logic to filter only relevant files (Pre/Post/Masks)
                # This is a simplification.
                
                local_path = os.path.join(target_dir, os.path.relpath(key, prefix))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if not os.path.exists(local_path):
                    print(f"Downloading {key} to {local_path}...")
                    s3.download_file(bucket_name, key, local_path)
                else:
                    print(f"Skipping {key}, already exists.")
                    
                count += 1
                if count >= limit:
                    print("Sample download limit reached.")
                    return
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == '__main__':
    download_sn8_sample()
