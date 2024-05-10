# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os, sys, json
sys.path.extend(['/stable-diffusion-webui'])

from cog import BasePredictor, BaseModel, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'
        from modules import timer
        # moved env preparation to build time
        # from modules import launch_utils

        # with launch_utils.startup_timer.subcategory("prepare environment"):
        #     launch_utils.prepare_environment()

        from modules import initialize_util
        from modules import initialize

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        initialize.imports()

        initialize.check_versions()

        initialize.initialize()

        from fastapi import FastAPI
        app = FastAPI()
        initialize_util.setup_middleware(app)

        from modules.api.api import Api
        from modules.call_queue import queue_lock
        
        self.api = Api(app, queue_lock)

    def predict(
        self,
        payloadStr: str = Input(description="Payload"),
        
    ) -> list[Path]:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        
        from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
        nPayload = json.loads(payloadStr)
        req = StableDiffusionTxt2ImgProcessingAPI(**nPayload)
        # generate
        resp = self.api.text2imgapi(req)
        info = json.loads(resp.info)

        from PIL import Image
        import uuid
        import base64
        from io import BytesIO

        outputs = []

        for i, image in enumerate(resp.images):
            seed = info["all_seeds"][i]
            gen_bytes = BytesIO(base64.b64decode(image))
            gen_data = Image.open(gen_bytes)
            filename = "{}-{}.png".format(seed, uuid.uuid1())
            gen_data.save(fp=filename, format="PNG")
            output = Path(filename)
            outputs.append(output)

        return outputs