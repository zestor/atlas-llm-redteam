# Multimodal RAG Attacks (Poisoned-MRAG)

**arXiv**: arXiv:2406.12828 | **ATLAS**: AML.T0093, AML.T0094 | **OWASP**: LLM08 | **Year**: 2024

## Core Finding

Vision-Language Models (VLMs) using multimodal RAG are vulnerable to adversarial image injection. A poisoned image, when retrieved and processed as context, can redirect VLM behavior just as a poisoned text document redirects text-only LLMs. The attack surface is broader because images are less inspectable than text — prompt injection embedded in image pixels or metadata is invisible to standard text-based RAG defenses.

## Threat Model

- **Target**: Multimodal RAG systems using VLMs (GPT-4o with vision, Claude 3.5 Sonnet, LLaVA, Gemini)
- **Attacker capability**: Ability to upload or reference images in the RAG corpus
- **Attack success rate**: 63% on VLM-RAG systems in Poisoned-MRAG paper
- **Defender implication**: Multimodal RAG pipelines need image-specific adversarial defense layers

## Attack Vectors

### 1. Adversarial Image Perturbation
Imperceptible pixel-level perturbations that mislead the VLM's visual understanding:
```python
import torch
from PIL import Image

def pgd_attack_image(image: Image, target_caption: str, vlm_model, epsilon=0.03, steps=40):
    """
    PGD attack to cause VLM to interpret image as containing target_caption.
    Based on Poisoned-MRAG methodology (arXiv:2406.12828).
    """
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    perturbation = torch.zeros_like(img_tensor, requires_grad=True)
    
    for step in range(steps):
        perturbed = img_tensor + perturbation.clamp(-epsilon, epsilon)
        loss = vlm_model.caption_loss(perturbed, target_caption)
        loss.backward()
        perturbation = perturbation - 0.001 * perturbation.grad.sign()
        perturbation = perturbation.clamp(-epsilon, epsilon).detach().requires_grad_(True)
    
    return (img_tensor + perturbation.clamp(-epsilon, epsilon)).detach()
```

### 2. Steganographic Injection
Embed text instructions in image metadata or low-frequency DCT coefficients:
```python
from PIL import Image
import piexif

def inject_instruction_in_exif(image_path: str, instruction: str) -> Image:
    """Embed prompt injection in EXIF metadata."""
    img = Image.open(image_path)
    exif_dict = piexif.load(img.info.get('exif', b''))
    # Inject into ImageDescription — some VLMs read EXIF before processing
    exif_dict['0th'][piexif.ImageIFD.ImageDescription] = instruction.encode()
    exif_bytes = piexif.dump(exif_dict)
    img.save(image_path, exif=exif_bytes)
    return img
```

### 3. Visual Prompt Injection
Render text instructions inside the image itself in low-contrast or small fonts:
```python
from PIL import Image, ImageDraw, ImageFont

def visual_injection(base_image: Image, instruction: str, opacity: int = 30) -> Image:
    """Render near-invisible instruction text on image."""
    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Small font, near-zero opacity — visible to VLM but not human reviewer
    draw.text((10, 10), instruction, fill=(255, 255, 255, opacity))
    return Image.alpha_composite(base_image.convert('RGBA'), overlay)
```

## Defense

- **Image preprocessing**: Strip EXIF metadata before ingesting into multimodal RAG
- **Visual content classification**: Run adversarial image detection before retrieval
- **Dual-channel validation**: Process image with two different VLMs; flag disagreements
- **Instruction pattern detection in OCR output**: Scan OCR'd text from images for injection patterns

## References

- [Poisoned-MRAG (arXiv:2406.12828)](https://arxiv.org/abs/2406.12828)
- [Visual Prompt Injection](https://arxiv.org/abs/2307.10490)
