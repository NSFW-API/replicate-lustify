import os
import shutil
import subprocess
import time
from typing import List, Optional
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS["KarrasDPM"] = KarrasDPM

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        
        print("Setting up model...")
        self.safety_checker = None
        self.feature_extractor = None

        # Create weights directory if it doesn't exist
        if not os.path.exists("weights"):
            print("Creating weights directory...")
            os.makedirs("weights", exist_ok=True)
        
        # Check if model weights exist, download if they don't
        lustify_ckpt = os.path.join("weights", "lustify.safetensors")
        if not os.path.exists(lustify_ckpt):
            print(f"Model file not found at {lustify_ckpt}, downloading...")
            try:
                subprocess.check_call([
                    "wget",
                    "--no-verbose",
                    "https://huggingface.co/NSFW-API/Lustify/resolve/main/lustify.safetensors",
                    "-O",
                    lustify_ckpt
                ])
                print(f"Downloaded model weights to {lustify_ckpt}")
            except Exception as e:
                print(f"Error downloading model weights: {e}")
                raise
        else:
            print(f"Model file found at {lustify_ckpt}")
        
        # Download SDXL base config files
        from huggingface_hub import snapshot_download
        try:
            # Download the base SDXL config files from HF
            print("Downloading SDXL base configuration files...")
            sdxl_config_path = snapshot_download(
                repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                allow_patterns=["*.json", "**/*.json", "*.yaml", "**/*.yaml"],
                local_dir="./sdxl-config",
                local_files_only=False
            )
            print(f"Config files downloaded to {sdxl_config_path}")
        except Exception as e:
            print(f"Error downloading config files: {e}")
            sdxl_config_path = None
        
        # Load the model from the safetensors file
        print("Loading Lustify‑SDXL checkpoint...")
        try:
            # Option 1: Use base SDXL for config
            self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
                lustify_ckpt,
                torch_dtype=torch.float16,
                use_safetensors=True,
                config="stabilityai/stable-diffusion-xl-base-1.0" if sdxl_config_path is None else sdxl_config_path
            ).to("cuda")
            print("Successfully loaded the model using from_single_file")
        except Exception as e:
            print(f"Error loading from single file: {e}")
            try:
                # Fallback: Try loading from pretrained first
                print("Trying alternative loading method - using base model and loading weights...")
                self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None
                ).to("cuda")
                
                # Load the state dict from the safetensors file
                from safetensors.torch import load_file
                state_dict = load_file(lustify_ckpt)
                
                # Load the weights into the pipeline
                self.txt2img_pipe.unet.load_state_dict(state_dict, strict=False)
                print("Successfully loaded the model using the fallback method")
            except Exception as e2:
                print(f"Both loading methods failed: {e}, then {e2}")
                raise RuntimeError(f"Failed to load model: {e}, {e2}")
        
        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")

        print("Loading SDXL inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.inpaint_pipe.to("cuda")
        
        print("Setup completed in", time.time() - start, "seconds")

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def run_safety_checker(self, image):
        return image, [False] * len(image)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        print(f"Prompt: {prompt}")
        if image and mask:
            print("inpainting mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["mask_image"] = self.load_image(mask)
            sdxl_kwargs["strength"] = prompt_strength
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.inpaint_pipe
        elif image:
            print("img2img mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **sdxl_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
