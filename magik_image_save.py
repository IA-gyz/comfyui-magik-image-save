"""
Magik Image Save - Optimized standalone image saving node for ComfyUI
Based on WAS Node Suite's WAS_Image_Save with significant performance optimizations

Optimizations implemented:
1. Fast tensor-to-PIL conversion using memory-efficient numpy operations
2. Format-optimized encoding parameters
3. Lazy metadata initialization
4. Reduced filesystem overhead
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy_api.latest import io, ui


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.webp', '.bmp')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
DEFAULT_OUTPUT_DIR = os.path.normpath(DEFAULT_OUTPUT_DIR)

# Fast WEBP encoding method (0=fastest, 6=slowest/best). We use 4 for balance.
WEBP_METHOD = 4
WEBP_QUALITY = 92


# ============================================================================
# LIGHTWEIGHT LOGGING
# ============================================================================

class _Colors:
    END = '\33[0m'
    BLUE = '\33[34m'
    RED = '\33[31m'
    YELLOW = '\33[33m'

def log_msg(msg: str):
    print(f"{_Colors.BLUE}Magik Image Save:{_Colors.END} {msg}")

def log_err(msg: str):
    print(f"{_Colors.RED}Magik Image Save Error:{_Colors.END} {msg}")

def log_warn(msg: str):
    print(f"{_Colors.YELLOW}Magik Image Save Warning:{_Colors.END} {msg}")


# ============================================================================
# FAST TOKEN PARSER
# ============================================================================

class FastTokens:
    """Lightweight token parser"""

    def __init__(self):
        self.tokens = {
            '[time]': str(int(time.time())),
        }
        self._hostname: Optional[str] = None
        self._user: Optional[str] = None

    @property
    def hostname(self) -> str:
        if self._hostname is None:
            import socket
            self._hostname = socket.gethostname()
        return self._hostname

    @property
    def user(self) -> str:
        if self._user is None:
            try:
                self._user = os.getlogin() or 'null'
            except Exception:
                self._user = 'null'
        return self._user

    def parse(self, text: str) -> str:
        """Parse tokens in text"""
        if '[time]' in text:
            text = text.replace('[time]', self.tokens['[time]'])

        if '[time(' in text:
            import time as time_mod
            def _repl(m):
                fmt = m.group(1)
                return time_mod.strftime(fmt, time_mod.localtime())
            text = re.sub(r'\[time\((.*?)\)\]', _repl, text)

        if '[hostname]' in text:
            text = text.replace('[hostname]', self.hostname)

        if '[user]' in text:
            text = text.replace('[user]', self.user)

        return text


# ============================================================================
# SUBFOLDER VALIDATION
# ============================================================================

def is_valid_subfolder(subfolder: str) -> bool:
    """
    True if `subfolder` is safe to join onto an output directory: relative
    (no leading slash, no drive letter), free of '.'/'..' traversal
    components, and free of characters invalid in folder names on common
    filesystems. Kept conservative across Windows/Linux/macOS since ComfyUI
    runs on all three.
    """
    normalized = subfolder.replace('\\', '/')

    if normalized.startswith('/'):
        return False
    if re.match(r'^[a-zA-Z]:', normalized):  # Windows drive letter, e.g. "C:"
        return False

    parts = [p for p in normalized.split('/') if p != '']
    if not parts:
        return False

    invalid_chars = re.compile(r'[<>:"|?*\x00-\x1f]')
    for part in parts:
        if part in ('.', '..'):
            return False
        if invalid_chars.search(part):
            return False
        if part != part.strip():
            return False

    return True


# ============================================================================
# ROBUST TENSOR TO PIL CONVERSION (FIXED)
# ============================================================================

def fast_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """
    More robust tensor to PIL conversion (fixes WebP issues with large images).
    """
    if image.ndim == 4:
        image = image[0]  # Take first image in batch

    # Ensure float32 and [0,1] range
    if image.dtype != torch.float32:
        image = image.float()
    image = image.clamp(0.0, 1.0)

    if image.is_cuda:
        np_img = image.detach().cpu().numpy()
    else:
        np_img = image.numpy()

    # Safer conversion
    np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)

    if np_img.shape[2] == 3:
        return Image.fromarray(np_img, mode='RGB')
    elif np_img.shape[2] == 4:
        return Image.fromarray(np_img, mode='RGBA')
    else:
        return Image.fromarray(np_img)


# ============================================================================
# OPTIMIZED IMAGE SAVER
# ============================================================================

class MagikImageSaver:
    """High-performance image saving with format-specific optimizations"""

    SAVE_KWARGS = {
        'png': {
            'compress_level': 3,
        },
        'webp': {
            'quality': WEBP_QUALITY,
            'method': WEBP_METHOD,
        },
        'jpeg': {
            'quality': 95,
            'optimize': False,
        },
        'jpg': {
            'quality': 95,
            'optimize': False,
        },
    }

    @classmethod
    def save(cls, img: Image.Image, filepath: str, extension: str,
             pnginfo: Optional[PngInfo] = None, dpi: int = 96,
             quality: int = 100, lossless_webp: bool = False) -> bool:
        """
        Save image with format-optimized parameters.
        Returns True on success, False on failure.
        """
        ext = extension.lower().lstrip('.')

        try:
            if ext == 'png':
                kwargs = cls.SAVE_KWARGS['png'].copy()
                if pnginfo is not None:
                    kwargs['pnginfo'] = pnginfo
                img.save(filepath, **kwargs)

            elif ext in ('jpg', 'jpeg'):
                kwargs = cls.SAVE_KWARGS['jpg'].copy()
                kwargs['quality'] = quality
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                kwargs['dpi'] = (dpi, dpi)
                img.save(filepath, **kwargs)

            elif ext == 'webp':
                kwargs = cls.SAVE_KWARGS['webp'].copy()
                kwargs['quality'] = quality
                kwargs['lossless'] = lossless_webp
                
                # Convert RGBA to RGB for lossy WebP to prevent alpha channel mangling
                if img.mode == 'RGBA' and not lossless_webp:
                    img = img.convert('RGB')
                elif img.mode not in ('RGB', 'RGBA', 'L'):
                    img = img.convert('RGB')
                
                img.save(filepath, **kwargs)

            elif ext == 'gif':
                img.save(filepath, save_all=True)

            elif ext == 'tiff':
                img.save(filepath, quality=quality)

            elif ext == 'bmp':
                img.save(filepath)

            else:
                img.save(filepath)

            return True

        except Exception as e:
            log_err(f"Failed to save {filepath}: {e}")
            return False


# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class Magik_Image_Save(io.ComfyNode):
    """
    High-performance image saving node with significant optimizations.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magik.ImageSave",
            display_name="Magik Image Save \U0001F680",
            category="Magik/IO",
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("output_path", default='[time(%Y-%m-%d)]'),
                io.String.Input("subfolder", default=""),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.String.Input("filename_delimiter", default="_"),
                io.Int.Input("filename_number_padding", default=4, min=1, max=9, step=1),
                io.Combo.Input("filename_number_start", options=["false", "true"]),
                io.Combo.Input("extension", options=['png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp']),
                io.Int.Input("dpi", default=96, min=1, max=2400, step=1),
                io.Int.Input("quality", default=92, min=1, max=100, step=1),
                io.Combo.Input("lossless_webp", options=["false", "true"]),
                io.Combo.Input("overwrite_mode", options=["false", "prefix_as_filename"]),
                io.Combo.Input("show_previews", options=["true", "false"], optional=True),
                io.Boolean.Input("save", default=True, optional=True),
            ],
            outputs=[
                io.Image.Output("images", display_name="images"),
                io.String.Output("files", display_name="files"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, images, output_path='', subfolder='', filename_prefix="ComfyUI",
                filename_delimiter='_', extension='png', dpi=96, quality=92,
                lossless_webp="false", overwrite_mode='false', filename_number_padding=4,
                filename_number_start='false', show_previews="true", save=True):
        # Saving disabled: pass images through, touch nothing on disk
        if not save:
            return io.NodeOutput(images, "", ui=ui.SavedImages([]))

        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo

        lossless_webp = lossless_webp == "true"
        embed_workflow = prompt is not None or extra_pnginfo is not None
        show_previews = show_previews == "true"
        start_at_front = filename_number_start == "true"

        tokens = FastTokens()
        filename_prefix = tokens.parse(filename_prefix)

        if not output_path or output_path in ('none', '.', ''):
            output_path = DEFAULT_OUTPUT_DIR
        else:
            output_path = tokens.parse(output_path)
            if not os.path.isabs(output_path):
                output_path = os.path.join(DEFAULT_OUTPUT_DIR, output_path)

        output_path = os.path.normpath(output_path)

        subfolder = subfolder.strip() if subfolder else ''
        if subfolder:
            subfolder = tokens.parse(subfolder)
            if is_valid_subfolder(subfolder):
                output_path = os.path.normpath(os.path.join(output_path, subfolder))
            else:
                log_warn(f"Ignoring invalid subfolder '{subfolder}'")

        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            log_err(f"Cannot create directory {output_path}: {e}")
            return io.NodeOutput(images, "", ui=ui.SavedImages([]))

        file_ext = f".{extension}"
        if file_ext not in ALLOWED_EXT:
            log_warn(f"Extension '{extension}' not in allowed list, using png")
            file_ext = ".png"
            extension = "png"

        pnginfo = None
        if extension == 'png' and embed_workflow:
            pnginfo = PngInfo()
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))

        if overwrite_mode == 'prefix_as_filename':
            counter = None
        else:
            counter = cls._find_next_counter(
                output_path, filename_prefix, filename_delimiter,
                filename_number_padding, start_at_front
            )

        results = []
        output_files = []

        for image in images:
            img = fast_tensor_to_pil(image)

            # Extra safety: for lossy WebP, ensure RGB (conversion also done in saver)
            if extension == 'webp' and not lossless_webp and img.mode == 'RGBA':
                img = img.convert('RGB')

            if overwrite_mode == 'prefix_as_filename':
                filename = f"{filename_prefix}{file_ext}"
            else:
                if start_at_front:
                    filename = f"{counter:0{filename_number_padding}}{filename_delimiter}{filename_prefix}{file_ext}"
                else:
                    filename = f"{filename_prefix}{filename_delimiter}{counter:0{filename_number_padding}}{file_ext}"

                filepath = os.path.join(output_path, filename)
                while os.path.exists(filepath):
                    counter += 1
                    if start_at_front:
                        filename = f"{counter:0{filename_number_padding}}{filename_delimiter}{filename_prefix}{file_ext}"
                    else:
                        filename = f"{filename_prefix}{filename_delimiter}{counter:0{filename_number_padding}}{file_ext}"
                    filepath = os.path.join(output_path, filename)

            filepath = os.path.join(output_path, filename)

            success = MagikImageSaver.save(
                img, filepath, extension,
                pnginfo=pnginfo, dpi=dpi,
                quality=quality, lossless_webp=lossless_webp
            )

            if success:
                output_files.append(filepath)
                log_msg(f"Saved: {filepath}")

                if show_previews:
                    subfolder = cls._get_subfolder(filepath, DEFAULT_OUTPUT_DIR)
                    results.append(ui.SavedResult(
                        filename=filename,
                        subfolder=subfolder,
                        type=io.FolderType.output,
                    ))

                if counter is not None:
                    counter += 1

        return io.NodeOutput(images, ",".join(output_files), ui=ui.SavedImages(results))

    @staticmethod
    def _get_subfolder(filepath: str, base_output: str) -> str:
        try:
            rel = os.path.relpath(os.path.dirname(filepath), base_output)
            if rel == '.':
                return ""
            return rel.replace('\\', '/')
        except ValueError:
            return ""

    @staticmethod
    def _find_next_counter(directory: str, prefix: str, delimiter: str,
                           padding: int, start_at_front: bool = False) -> int:
        """Find next available counter"""
        try:
            listing = os.listdir(directory)
        except OSError:
            return 1

        pattern = re.compile(
            rf"^{re.escape(prefix)}{re.escape(delimiter)}(\d{{{padding}}})" if not start_at_front 
            else rf"^(\d{{{padding}}}){re.escape(delimiter)}{re.escape(prefix)}"
        )

        max_counter = 0
        for filename in listing:
            m = pattern.match(filename)
            if m:
                try:
                    num = int(m.group(1))
                    if num > max_counter:
                        max_counter = num
                except ValueError:
                    continue

        return max_counter + 1


# ============================================================================
# SIMPLIFIED SINGLE-IMAGE NODE
# ============================================================================

class Magik_Image_Save_Simple(io.ComfyNode):
    """
    Simplified clone of Magik_Image_Save with a reduced input set for
    single-image saving. Duplicate filenames are resolved automatically by
    appending an incrementing number, rather than exposing overwrite/numbering
    controls.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magik.ImageSaveSimple",
            display_name="Magik Image Save Simple \U0001F680",
            category="Magik/IO",
            is_output_node=True,
            inputs=[
                io.Image.Input("image"),
                io.String.Input("output_path", default='[time(%Y-%m-%d)]'),
                io.String.Input("subfolder", default=""),
                io.String.Input("filename", default="ComfyUI"),
                io.Combo.Input("extension", options=['png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp']),
                io.Int.Input("dpi", default=96, min=1, max=2400, step=1),
                io.Int.Input("quality", default=92, min=1, max=100, step=1),
                io.Combo.Input("lossless_webp", options=["false", "true"]),
                io.Combo.Input("show_preview", options=["true", "false"], optional=True),
                io.Boolean.Input("save", default=True, optional=True),
            ],
            outputs=[
                io.Image.Output("image", display_name="image"),
                io.String.Output("file", display_name="file"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, image, output_path='', subfolder='', filename="ComfyUI", extension='png',
                dpi=96, quality=92, lossless_webp="false", show_preview="true", save=True):
        # Saving disabled: pass image through, touch nothing on disk
        if not save:
            return io.NodeOutput(image, "", ui=ui.SavedImages([]))

        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo

        lossless_webp = lossless_webp == "true"
        embed_workflow = prompt is not None or extra_pnginfo is not None
        show_preview = show_preview == "true"

        tokens = FastTokens()
        filename = tokens.parse(filename)

        if not output_path or output_path in ('none', '.', ''):
            output_path = DEFAULT_OUTPUT_DIR
        else:
            output_path = tokens.parse(output_path)
            if not os.path.isabs(output_path):
                output_path = os.path.join(DEFAULT_OUTPUT_DIR, output_path)

        output_path = os.path.normpath(output_path)

        subfolder = subfolder.strip() if subfolder else ''
        if subfolder:
            subfolder = tokens.parse(subfolder)
            if is_valid_subfolder(subfolder):
                output_path = os.path.normpath(os.path.join(output_path, subfolder))
            else:
                log_warn(f"Ignoring invalid subfolder '{subfolder}'")

        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            log_err(f"Cannot create directory {output_path}: {e}")
            return io.NodeOutput(image, "", ui=ui.SavedImages([]))

        file_ext = f".{extension}"
        if file_ext not in ALLOWED_EXT:
            log_warn(f"Extension '{extension}' not in allowed list, using png")
            file_ext = ".png"
            extension = "png"

        pnginfo = None
        if extension == 'png' and embed_workflow:
            pnginfo = PngInfo()
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))

        img = fast_tensor_to_pil(image)

        # Extra safety: for lossy WebP, ensure RGB (conversion also done in saver)
        if extension == 'webp' and not lossless_webp and img.mode == 'RGBA':
            img = img.convert('RGB')

        # Auto-number on collision: filename.ext, then filename_1.ext, filename_2.ext, ...
        filepath = os.path.join(output_path, f"{filename}{file_ext}")
        dupe_counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(output_path, f"{filename}_{dupe_counter}{file_ext}")
            dupe_counter += 1

        success = MagikImageSaver.save(
            img, filepath, extension,
            pnginfo=pnginfo, dpi=dpi,
            quality=quality, lossless_webp=lossless_webp
        )

        if not success:
            return io.NodeOutput(image, "", ui=ui.SavedImages([]))

        log_msg(f"Saved: {filepath}")

        if show_preview:
            subfolder = Magik_Image_Save._get_subfolder(filepath, DEFAULT_OUTPUT_DIR)
            results = [ui.SavedResult(
                filename=os.path.basename(filepath),
                subfolder=subfolder,
                type=io.FolderType.output,
            )]
            return io.NodeOutput(image, filepath, ui=ui.SavedImages(results))
        else:
            return io.NodeOutput(image, filepath, ui=ui.SavedImages([]))
