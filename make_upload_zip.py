import os
import zipfile

# Define what to include
include_files = [
    'app.py',
    'inference.py',
    'unet_autoencoder.py',
    'unet_epoch_30.pth',
    'requirements.txt',
    'README.md',
]

include_dirs = [
    'templates',
    'static',
]

def zip_project(output_zip='project_upload.zip'):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add files
        for file in include_files:
            if os.path.exists(file):
                print(f'Adding file: {file}')
                zipf.write(file)
            else:
                print(f'File not found (skipped): {file}')
        # Add directories
        for folder in include_dirs:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, start=os.path.dirname(folder))
                        print(f'Adding file: {filepath}')
                        zipf.write(filepath, arcname)
            else:
                print(f'Folder not found (skipped): {folder}')
    print(f'\nCreated zip file: {output_zip}')

if __name__ == '__main__':
    zip_project()
