import os
import yaml

# Replace this with the path to the parent folder containing the labelme-generated folders
parent_folder = 'D:\All_Codes\MaskRCNNTest\Mask_RCNN\samples\plant2227_dataset\\val_data\labelme_json'

# Define the content of the info.yaml file as a Python dictionary
info_yaml_content = {
    'label_names': ['_background_', 'main_stem'],  # Update this list with your label names
}

# Iterate through the subfolders in the parent folder
for subdir, dirs, files in os.walk(parent_folder):
    if 'label.png' in files:
        # Generate the info.yaml file
        info_yaml_path = os.path.join(subdir, 'info.yaml')
        with open(info_yaml_path, 'w') as info_yaml_file:
            yaml.dump(info_yaml_content, info_yaml_file)
        print(f'Generated info.yaml file for folder: {subdir}')
