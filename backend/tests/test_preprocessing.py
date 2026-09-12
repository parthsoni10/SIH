import io
import numpy as np
from PIL import Image
from app.modules.preprocessing import preprocess_image, PreprocessedImage

def create_synthetic_image_bytes(width=800, height=600, color=(100, 150, 200), fmt="JPEG"):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def test_preprocess_synthetic_jpeg():
    bytes_data = create_synthetic_image_bytes(1000, 500)
    result = preprocess_image(bytes_data, max_edge=2000)
    
    assert isinstance(result, PreprocessedImage)
    assert result.original_size == (1000, 500)
    assert isinstance(result.image, np.ndarray)
    assert result.image.shape[0] == 500
    assert result.image.shape[1] == 1000

def test_preprocess_resizing():
    bytes_data = create_synthetic_image_bytes(3000, 1500)
    result = preprocess_image(bytes_data, max_edge=1000)
    
    assert result.original_size == (3000, 1500)
    assert result.processed_size == (1000, 500)
    assert max(result.image.shape[:2]) == 1000

def test_empty_bytes_raises():
    raised = False
    try:
        preprocess_image(b"")
    except ValueError:
        raised = True
    assert raised is True
