
from .hunyuan_3d_node import install_check, Hunyuan3DImageTo3DMesh, Hunyuan3DTexture3DMesh

install_check()

NODE_CLASS_MAPPINGS = {
    "Hunyuan3DImageTo3DMesh": Hunyuan3DImageTo3DMesh,
    "Hunyuan3DTexture3DMesh": Hunyuan3DTexture3DMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DImageTo3DMesh": "Hunyuan3D-2 Image to 3D Mesh",
    "Hunyuan3DTexture3DMesh": "Hunyuan3D-2 Texture 3D Mesh"
}
