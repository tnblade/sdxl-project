# service/tasks.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import cả 3 công cụ
from core.text_to_image import generate_text_image
from core.image_to_image import img2img_edit
from core.controlnet import generate_controlnet_image
from config import DEFAULT_WIDTH, DEFAULT_HEIGHT

def run_generation_task(
    prompt: str, 
    negative_prompt: str = "low quality", 
    width=DEFAULT_WIDTH, 
    height=DEFAULT_HEIGHT, 
    steps=20, 
    seed=-1,
    # Thêm các tham số mới để nhận diện chế độ
    image_path: str = None,     # Dùng cho Img2Img hoặc ControlNet
    task_type: str = "txt2img", # txt2img | img2img | controlnet
    control_type: str = None,   # canny | depth | openpose
    strength: float = 0.7       # Độ mạnh cho Img2Img
):
    print(f"🔨 [TASK STARTED] Type: {task_type} | Prompt: {prompt[:30]}...")
    
    try:
        seed = int(seed) if int(seed) != -1 else None
        output_path = ""

        # --- LOGIC RẼ NHÁNH ---
        if task_type == "img2img" and image_path:
            # Chạy Image to Image
            output_path = img2img_edit(
                init_image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=steps,
                seed=seed
            )
            
        elif task_type == "controlnet" and image_path:
            # Chạy ControlNet
            output_path = generate_controlnet_image(
                prompt=prompt,
                control_image_path=image_path,
                control_type=control_type,
                negative_prompt=negative_prompt,
                width=width, 
                height=height,
                steps=steps,
                seed=seed
            )
            
        else:
            # Mặc định: Text to Image
            output_path = generate_text_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                seed=seed
            )
        
        print(f"✅ [TASK DONE] Saved at: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ [TASK FAILED] Error: {e}")
        raise e