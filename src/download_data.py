import os
import glob
import boto3
from botocore import UNSIGNED
from botocore.client import Config

def download_sn8_sample(target_dir='data/SN8'):
    """
    Downloads a sample of SpaceNet-8 data from S3.
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'spacenet-dataset'
    prefix_base = 'spacenet/SN8_floods/Germany_Training_Public/'
    
    # 1. Gather existing Pre-event Tile IDs
    pre_dir = os.path.join(target_dir, 'Germany_Training_Public', 'PRE-event')
    existing_pre = glob.glob(os.path.join(pre_dir, '*.tif'))
    target_suffixes = set()
    for p in existing_pre:
        parts = os.path.basename(p).split('_')
        suffix = '_'.join(parts[1:])
        target_suffixes.add(suffix)
        
    print(f"Found {len(target_suffixes)} existing Pre-event tiles. Downloading matches...")
    
    subfolders = ['POST-event', 'annotations'] 
    
    from concurrent.futures import ThreadPoolExecutor
    
    def download_worker(args):
        bucket, key, local_path, s3_client = args
        try:
            if not os.path.exists(local_path):
                # atomic check might fail but ok for now
                # print(f"Downloading {key}...")
                s3_client.download_file(bucket, key, local_path)
                return 1
            return 0
        except Exception as e:
            print(f"Failed {key}: {e}")
            return 0

    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED)) # pass client or create new? thread safe? 
    # boto3 client is thread safe.

    download_tasks = []

    for sub in subfolders:
        prefix = prefix_base + sub + '/'
        print(f"Listing and preparing tasks for {sub}...")
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue
                    
                    filename = os.path.basename(key)
                    should_download = False
                    
                    if sub == 'annotations':
                        tile_id = filename.replace('.geojson', '')
                        if (tile_id + '.tif') in target_suffixes:
                            should_download = True
                    else:
                        for s in target_suffixes:
                            if filename.endswith(s):
                                should_download = True
                                break
                    
                    if should_download:
                        local_path = os.path.join(target_dir, os.path.relpath(key, 'spacenet/SN8_floods/'))
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        if not os.path.exists(local_path):
                             download_tasks.append((bucket_name, key, local_path, s3_client))
            
        except Exception as e:
            print(f"Error listing {sub}: {e}")

    print(f"Starting download of {len(download_tasks)} files with 16 threads...")
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(download_worker, download_tasks))
        
    print(f"Downloaded {sum(results)} files.")



if __name__ == '__main__':
    download_sn8_sample()
