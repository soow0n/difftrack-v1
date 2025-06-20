from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_cogvideox"] = ["CogVideoXPipeline"]
    _import_structure["pipeline_cogvideox_pag"] = ["CogVideoXPipeline_PAG"]
    _import_structure["pipeline_cogvideox_fun_control"] = ["CogVideoXFunControlPipeline"]
    _import_structure["pipeline_cogvideox_image2video"] = ["CogVideoXImageToVideoPipeline"]
    _import_structure["pipeline_cogvideox_image2video_2b"] = ["CogVideoXImageToVideoPipeline2B"]
    _import_structure["pipeline_cogvideox_image2video_2b_tracking"] = ["CogVideoXImageToVideoTrackPipeline2B"]
    _import_structure["pipeline_cogvideox_video2video"] = ["CogVideoXVideoToVideoPipeline"]
    _import_structure["pipeline_cogvideox_inversion"] = ["CogVideoXInversePipeline"]
    _import_structure["pipeline_cogvideox_tracking"] = ["CogVideoXTrackPipeline"]
    _import_structure["pipeline_cogvideox_image2video_tracking"] = ["CogVideoXImageToVideoTrackPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_cogvideox import CogVideoXPipeline
        from .pipeline_cogvideox_pag import CogVideoXPipeline_PAG
        from .pipeline_cogvideox_fun_control import CogVideoXFunControlPipeline
        from .pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
        from .pipeline_cogvideox_image2video_2b import CogVideoXImageToVideoPipeline2B
        from .pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline
        from .pipeline_cogvideox_inversion import CogVideoXInversePipeline
        from .pipeline_cogvideox_tracking import CogVideoXTrackPipeline
        from .pipeline_cogvideox_image2video_2b_tracking import CogVideoXImageToVideoTrackPipeline2B
        from .pipeline_cogvideox_image2video_tracking import CogVideoXImageToVideoTrackPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
