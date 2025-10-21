from dataclasses import dataclass
import viser
import copy
import torch
from pathlib import Path
from nerfview import CameraState, RenderTabState
from sklearn.decomposition import PCA
from typing import List, Union, Literal, Tuple
from nerfview import Viewer, RenderTabState
from .viewer import ViewerState
import numpy as np
import torch
import matplotlib.cm as cm
from matplotlib.colors import CSS4_COLORS, to_hex, to_rgb
from .hierachical_primitive import HierachicalPrimitive
from tqdm import tqdm

@dataclass
class HierachicalViewerState(RenderTabState):
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal["RGB", "Feature"] = "RGB"
    layer_index: int = 0



class HierachicalViewer(Viewer):
    def __init__(self, server: viser.ViserServer, hierachical_primitive_path: Path, viewer_state: HierachicalViewerState, port=8080, with_feature: bool = False):
        self.port = port
        self.with_feature = with_feature
        self.viewer_state = viewer_state
        self.render_tab_state = copy.deepcopy(viewer_state)
        self.hierachical_primitive_path = hierachical_primitive_path
        self.hierachical_primitive_initialization()
        self.feature_pca()
        self.curret_renderer_cache = self.hierachical_primitive.get_renderer(0)
        super().__init__(server, self.render_function, None, 'rendering')
    
    def hierachical_primitive_initialization(self):
        "set up splat entity"
        self.hierachical_primitive = HierachicalPrimitive(with_feature=self.with_feature)
        self.hierachical_primitive.load_from_file(self.hierachical_primitive_path)

    def reset(self):
        self.render_tab_state = copy.deepcopy(self.viewer_state)
        self.hierachical_primitive_initialization()



    def feature_pca(self):
        if self.hierachical_primitive.with_feature is False:
            return
        else:
            self.feature_pcas: List[torch.Tensor] = []
            pca = PCA(n_components=3)
            for feature in tqdm(self.hierachical_primitive.source["feature"], desc="Feature PCA"):
                print(feature.shape)
                if feature.shape[0] <= 3:
                    print("Feature is too small, skipping")
                    zeros = torch.zeros(feature.shape[0], 3, dtype=feature.dtype, device=feature.device)
                    self.feature_pcas.append(zeros)
                    continue
                features_np = feature.cpu().numpy()
                features_np = features_np.reshape(features_np.shape[0], -1)
                features_pca = pca.fit_transform(features_np)
                features_pca = torch.from_numpy(features_pca).float()
                mins   = features_pca.min(dim=0).values    # shape (3,)
                maxs   = features_pca.max(dim=0).values    # shape (3,)
                ranges = maxs - mins
                eps    = 1e-8
                features_pca = (features_pca - mins) / (ranges + eps)
                self.feature_pcas.append(features_pca)
            self.hierachical_primitive.source["feature"] = self.feature_pcas



    def camera_state_parser(self, camera_state: CameraState, render_tab_state: RenderTabState)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w = camera_state.c2w
        width = render_tab_state.render_width
        height = render_tab_state.render_height
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to('cuda').unsqueeze(0)
        K = torch.from_numpy(K).float().to('cuda').unsqueeze(0)
        return K, c2w, width, height

    @torch.no_grad()
    def render_function(self, camera_state: CameraState, render_tab_state: ViewerState):
        state = self.render_tab_state
        K, c2w, width, height = self.camera_state_parser(camera_state, self.render_tab_state)
        if state.render_mode == "RGB":
            image = self.curret_renderer_cache.render(K, c2w, width, height, state.render_mode)
        elif state.render_mode == "Feature":
            image = self.curret_renderer_cache.render(K, c2w, width, height, state.render_mode)
        else:
            raise ValueError(f"Unsupported render mode: {state.render_mode}")
        return image.cpu().numpy()


    def _init_rendering_tab(self):
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")
    

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            self.feature_pca()


            self.render_mode_dropdown = server.gui.add_dropdown(
                "Render Mode",
                (
                    "RGB",
                    "Feature",
                ),
                initial_value=self.render_tab_state.render_mode,
                hint="Render mode to use.",
            )
            self.layer_index_slider = server.gui.add_slider(
                "Layer Index",
                initial_value=self.render_tab_state.layer_index,
                min=0,
                max=len(self.hierachical_primitive.source["geometry"]) - 1,
                step=1,
                hint="Layer index to render.",
            )

            @self.layer_index_slider.on_update
            def _(_) -> None:
                self.render_tab_state.layer_index = self.layer_index_slider.value
                del self.curret_renderer_cache
                torch.cuda.empty_cache()
                self.curret_renderer_cache = self.hierachical_primitive.get_renderer(self.render_tab_state.layer_index)
                self.rerender(_)

            @self.render_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.render_mode = self.render_mode_dropdown.value
                self.rerender(_)



        self._rendering_tab_handles.update(
            {
                "render_mode_dropdown": self.render_mode_dropdown,
                "layer_index_slider": self.layer_index_slider,
                "backgrounds_slider": self.render_tab_state.backgrounds,
            }
        )

        super()._populate_rendering_tab()
                





