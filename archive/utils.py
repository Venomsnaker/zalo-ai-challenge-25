from pathlib import Path
import re

def _extract_obj_name_pattern(folder_name: str):
    match = re.match(r"([a-zA-Z0-9]+)_\d+", folder_name)
    return match

def load_data(root_dir: str) -> list[dict]:
    data = []
    root_path = Path(root_dir)
    
    # Safety check
    if not root_path.is_dir():
        print(f"Error: Directory '{root_dir}' not found.")
        return []
    
    for item_folder in root_path.iterdir():
        if item_folder.is_dir():
            # Object name extraction
            folder_name = item_folder.name
            match = _extract_obj_name_pattern(folder_name)
            
            if not match:
                print(f"Warning: Skipping folder with unexpected name format: {folder_name}")
                continue
            obj_name = match.group(1)
            
            # Footage path extraction
            footage_path = item_folder / "drone_video.mp4"
            
            if not footage_path.exists():
                print(f"Warning: Skipping folder with no footage: {folder_name}")
                continue
            
            # Image paths extraction
            imgs_dir = item_folder / "object_images"
            img_paths = {}
            
            if imgs_dir.is_dir():
                for i, img_file in enumerate(sorted(imgs_dir.glob('img_*.jpg')), 1):
                    img_paths[f"image_{i}"] = str(img_file.as_posix())
            else:
                print(f"Warning: Skipping folder with no images directory: {folder_name}")
                continue
                
            # Compile the data
            item = {
                "id": folder_name,
                "object_name": obj_name,
                "footage_path": str(footage_path.as_posix()),
                "images": img_paths,
            }
            data.append(item)
    return data