import os
from typing import Dict, Any, List
import pandas as pd
from autogen_core.tools import FunctionTool
import base64


def preview_csv_head(path: str, n:int=100) -> Dict[str, Any]:
    """Preview the head of a CSV file."""
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File does not exist: {path}"}
    try:
        df = pd.read_csv(path)
        head = df.head(n).to_dict(orient="records")
        return {
            "ok": True,
            "path": path,
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
            "head": head,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

csv_head_tool = FunctionTool(
    func=preview_csv_head,
    name="preview_csv_head",
    description="""
    Preview first N rows of a CSSV and return row number, column number, column names and column types.
    """,
)


def image_to_base64(image_path: str) -> Dict[str, Any]:
    """
    Convert an image file to base64 encoding.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Dict[str, Any]: Dictionary containing success status, error message (if any), 
                       base64 encoded image data, and image metadata
    """
    if not os.path.exists(image_path):
        return {"ok": False, "error": f"Image file does not exist: {image_path}"}
    
    # Check if file is actually an image
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    if not image_path.lower().endswith(valid_extensions):
        return {"ok": False, "error": f"File is not a valid image type. Supported types: {valid_extensions}"}
    
    try:
        with open(image_path, "rb") as image_file:
            # Read the image file
            image_data = image_file.read()
            
            # Get the file size
            file_size = os.path.getsize(image_path)
            
            # Encode to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            # Get file extension for mime type
            file_extension = os.path.splitext(image_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(file_extension, 'image/jpeg')  # default to jpeg
            
            return {
                "ok": True,
                "path": image_path,
                "size": file_size,
                "mime_type": mime_type,
                "base64": encoded_image
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def gen_analytics_report(save_path: str, content: str):
    """
    Generate an analytics report and save it as a markdown file.
    Extract image paths from the markdown content, convert images to base64,
    and embed them directly in the markdown file.
    
    Args:
        save_path (str): Path where the markdown report should be saved
        content (str): The markdown content of the report
        
    Returns:
        Dict[str, Any]: Dictionary containing success status and error message (if any)
    """
    import re
    
    try:
        # Process the content to replace image paths with base64 encoded images
        processed_content = content
        
        # Find all image references in the markdown content
        # Looking for patterns like: ![alt text](path/to/image.png)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        image_matches = re.findall(image_pattern, content)
        
        # Replace each image path with base64 encoded image
        for alt_text, image_path in image_matches:
            # Clean the image path (remove any quotes)
            image_path = image_path.strip('"\'')
            
            # Convert image to base64
            image_data = image_to_base64(image_path)
            
            if image_data["ok"]:
                # Create base64 image string
                base64_img_str = f"data:{image_data['mime_type']};base64,{image_data['base64']}"
                
                # Replace the image reference with base64 encoded image
                old_ref = f'![{alt_text}]({image_path})'
                new_ref = f'![{alt_text}]({base64_img_str})'
                processed_content = processed_content.replace(old_ref, new_ref)
            # If there's an error with the image, we'll leave the original reference
        
        # Save the processed content to the file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
            
        return {
            "ok": True,
            "path": save_path
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

gen_analytics_report_tool = FunctionTool(
    func=gen_analytics_report,
    name="gen_analytics_report",
    description="""
    Generate an analytics report and save it as a markdown file.
    Extract image paths from the markdown content, convert images to base64,
    and embed them directly in the markdown file.
    """,
)