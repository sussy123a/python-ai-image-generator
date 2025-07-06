import tkinter as tk
from tkinter import ttk, messagebox
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk
import threading

class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        root.title("GAN Image Generator (Stable Diffusion)")
        root.geometry("600x700")

        self.prompt_label = ttk.Label(root, text="Enter prompt:")
        self.prompt_label.pack(pady=10)

        self.prompt_entry = ttk.Entry(root, width=60)
        self.prompt_entry.pack(pady=5)

        self.generate_button = ttk.Button(root, text="Generate Image", command=self.generate_image_thread)
        self.generate_button.pack(pady=10)

        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)

        self.status_label = ttk.Label(root, text="")
        self.status_label.pack(pady=5)

        self.pipe = None
        self.load_model()

    def load_model(self):
        self.status_label.config(text="Loading model, please wait...")
        self.root.update()

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None  # disables safety checker for speed (optional)
        )
        if torch.cuda.is_available():
            self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")

        self.status_label.config(text="Model loaded. Ready to generate!")

    def generate_image_thread(self):
        thread = threading.Thread(target=self.generate_image)
        thread.start()

    def generate_image(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a prompt.")
            return

        self.generate_button.config(state="disabled")
        self.status_label.config(text="Generating image, please wait...")
        self.root.update()

        try:
            # Use autocast for faster mixed precision on CUDA
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    image = self.pipe(
                        prompt,
                        num_inference_steps=50,
                        height=458,
                        width=458
                    ).images[0]
            else:
                # CPU fallback, no autocast
                image = self.pipe(
                    prompt,
                    num_inference_steps=49,
                    height=457,
                    width=457
                ).images[0]

            image.save("output.png")

            img = image.resize((256, 256))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            self.status_label.config(text="Image generated and saved as output.png")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Failed to generate image.")
        finally:
            self.generate_button.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()
