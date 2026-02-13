# unitok_image_processor.py
import torch
from torchvision.transforms import transforms
from PIL import Image

def normalize_01_into_pm1(x):
    """
    Normalize tensor values from [0, 1] range to [-1, 1] range.
    
    Args:
        x (`torch.Tensor`): Input tensor with values in [0, 1] range.
    
    Returns:
        `torch.Tensor`: Normalized tensor with values in [-1, 1] range.
    """
    return x * 2.0 - 1.0

unitok_preprocess = transforms.Compose([
    transforms.Resize(int(256 * 1.125)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize_01_into_pm1,
])

unitok_preprocess_tensor = transforms.Compose([
    transforms.Resize(int(256 * 1.125)),
    transforms.CenterCrop(256),
    lambda x: x / 255,
    normalize_01_into_pm1,
])

def get_pixel_values(image_path, device='cuda'):
    """
    Load and preprocess an image for UniTok model.
    
    This function loads an image from the specified path, applies UniTok-specific
    preprocessing (resize, center crop, normalization to [-1, 1] range), and
    returns the processed tensor on the specified device.
    
    Args:
        image_path (`str`):
            Path to the input image file.
        device (`str`, defaults to 'cuda'):
            Device to place the tensor on ('cuda' or 'cpu').
    
    Returns:
        `torch.Tensor`: Preprocessed image tensor of shape `[1, 3, 256, 256]`
            with values in [-1, 1] range, on the specified device.
    
    Example:
        ```python
        >>> pixel_values = get_pixel_values('/path/to/image.jpg')
        >>> print(pixel_values.shape)  # torch.Size([1, 3, 256, 256])
        >>> print(pixel_values.device)  # cuda:0
        ```
    """
    img = Image.open(image_path).convert("RGB")
    
    pixel_values = unitok_preprocess(img)
    
    pixel_values = pixel_values.unsqueeze(0).to(device)
    
    return pixel_values

def process_with_unitok(processor, images, text, **kwargs):
    """
    Process images and text using UniTok preprocessing for the vision component.
    
    This is a convenience function that combines UniTok image preprocessing with
    the standard text processing from a LlavaProcessor. It handles various input
    formats for images (paths, PIL Images, numpy arrays, tensors) and applies
    appropriate conversions.
    
    Args:
        processor:
            LlavaProcessor instance for text processing.
        images (`str`, `PIL.Image`, `numpy.ndarray`, `torch.Tensor`, or `list`, *optional*):
            Image or list of images to process. Can be file paths, PIL Images,
            numpy arrays, or torch tensors. If None, only text is processed.
        text (`str` or `list` of `str`):
            Text input(s) to process.
        **kwargs:
            Additional arguments to pass to the processor for text processing.
    
    Returns:
        `BatchFeature`: Dictionary-like object containing:
            - Text inputs from the processor (input_ids, attention_mask, etc.)
            - pixel_values: Preprocessed image tensor if images were provided
    
    Example:
        ```python
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> inputs = process_with_unitok(
        ...     processor, 
        ...     images="/path/to/image.jpg",
        ...     text="USER: <image>\nDescribe this image. ASSISTANT:"
        ... )
        >>> print(inputs.keys())  # dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
        ```
    """
    if images is not None:
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                import numpy as np
                if isinstance(img, np.ndarray):
                    if img.dtype in [np.float32, np.float64]:
                        img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            processed_img = unitok_preprocess(img)
            processed_images.append(processed_img)
        
        pixel_values = torch.stack(processed_images)
    
    text_inputs = processor(images=None, text=text, **kwargs)
    
    if images is not None:
        text_inputs['pixel_values'] = pixel_values
    
    return text_inputs


def process_with_unitok_diff(processor, images, text, **kwargs):  # [0, 1]
    if images is not None:
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            processed_img = unitok_preprocess_tensor(img)
            processed_images.append(processed_img)

        pixel_values = torch.stack(processed_images)

    text_inputs = processor(images=None, text=text, **kwargs)

    if images is not None:
        text_inputs['pixel_values'] = pixel_values

    return text_inputs