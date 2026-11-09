from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .magik_image_save import Magik_Image_Save, Magik_Image_Save_Simple


class MagikImageSaveExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Magik_Image_Save, Magik_Image_Save_Simple]


async def comfy_entrypoint() -> MagikImageSaveExtension:
    return MagikImageSaveExtension()
