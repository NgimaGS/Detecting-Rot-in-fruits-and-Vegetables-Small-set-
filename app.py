import os
import csv
from datetime import datetime
import customtkinter as ctk
import tkinter.filedialog as fd
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Import our custom engine
from pantry_engine import PantryEngine

# Global app settings configurations
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PantryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("Pantry Management & Freshness Detector")
        self.geometry("1100x700")

        # Initialize Data Engine
        self.engine = PantryEngine()
        self.image_path = None
        self.current_image = None
        self.results = None

        # --- Grid Layout (1x2) ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # 1. Left Sidebar Frame (width = 200)
        # ==========================================
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Branding / Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Pantry AI", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # Buttons
        self.upload_btn = ctk.CTkButton(self.sidebar_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=1, column=0, padx=20, pady=10)

        self.run_btn = ctk.CTkButton(self.sidebar_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.grid(row=2, column=0, padx=20, pady=10)

        self.export_btn = ctk.CTkButton(self.sidebar_frame, text="Export to CSV", command=self.export_csv)
        self.export_btn.grid(row=3, column=0, padx=20, pady=10)

        # ==========================================
        # 2. Main Right Frame
        # ==========================================
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Split vertical weights: Top 80%, Bottom 20%
        self.main_frame.grid_rowconfigure(0, weight=8) 
        self.main_frame.grid_rowconfigure(1, weight=2) 

        # Top: CTkLabel for Image Display
        self.image_label = ctk.CTkLabel(self.main_frame, text="No Image Uploaded", font=ctk.CTkFont(size=18))
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Bottom: CTkTextbox for Inventory Summary
        self.textbox = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(size=16))
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.textbox.insert("0.0", "Inventory Summary will appear here...\n")

    def upload_image(self):
        filetypes = (
            ('Images', '*.jpg *.jpeg *.png *.webp'),
            ('All files', '*.*')
        )
        filepath = fd.askopenfilename(title='Open Image', filetypes=filetypes)
        
        if filepath:
            self.image_path = filepath
            self.results = None
            
            # Display uploaded image un-annotated
            self.display_image(self.image_path)
            
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", f"Image successfully loaded: {os.path.basename(self.image_path)}\nReady for analysis.")

    def display_image(self, img_source):
        if isinstance(img_source, str):
            img = Image.open(img_source)
        else:
            img = img_source
            
        # Dynamically resize to fit the frame
        self.main_frame.update()
        frame_width = self.main_frame.winfo_width() - 20
        frame_height = int(self.main_frame.winfo_height() * 0.8) - 20
        
        # Fallback dimensions if unrendered
        if frame_width <= 0 or frame_height <= 0:
            frame_width, frame_height = 800, 500

        # Thumbnail respects aspect ratio
        img.thumbnail((frame_width, frame_height))
        
        # Create CustomTkinter Native Image
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        
        self.image_label.configure(image=ctk_img, text="")
        self.current_image = img

    def run_analysis(self):
        if not self.image_path:
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", "Action Blocked: Please upload an image first.\n")
            return

        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", "Processing image via ONNX PantryEngine...\n")
        self.update()

        try:
            # 1. Run Logic Engine
            self.results = self.engine.process_image(self.image_path)
            
            # 2. Draw Bounding Boxes via PIL
            orig_img = Image.open(self.image_path).convert("RGB")
            draw = ImageDraw.Draw(orig_img)
            
            # Color mapping for visual clarity
            color_map = {
                'FreshApple': '#00FF00', 'RottenApple': '#FF0000',
                'FreshBanana': '#FFFF00', 'RottenBanana': '#8B4513',
                'FreshOrange': '#FFA500', 'RottenOrange': '#800080'
            }

            try:
                # Load default UI font
                font = ImageFont.truetype("arial.ttf", 22)
            except IOError:
                font = ImageFont.load_default()

            for det in self.results["detections"]:
                label = det["label"]
                conf = det["confidence"]
                x1, y1, x2, y2 = det["box"]
                
                box_color = color_map.get(label, 'cyan')
                
                # Draw the main bounding box line
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
                
                # Format text
                text = f"{label} ({int(conf*100)}%)"

                # Draw label background
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((x1, y1), text, font=font)
                    draw.rectangle([bbox[0], bbox[1]-28, bbox[2]+8, bbox[3]], fill=box_color)
                    draw.text((x1+4, y1-26), text, fill="black", font=font)
                else:
                    draw.rectangle([x1, y1-25, x1+200, y1], fill=box_color)
                    draw.text((x1+4, y1-25), text, fill="black", font=font)

            # Update Canvas with annotated image
            self.display_image(orig_img)

            # 3. Print Output format to Textbox
            summary = "--- PANTRY INVENTORY SUMMARY ---\n\n"
            total = 0
            
            for item, count in self.results["inventory"].items():
                if count > 0:
                    summary += f"> {item}: {count}\n"
                    total += count
                    
            summary += "\n---------------------------------\n"
            summary += f"TOTAL DETECTED ITEMS: {total}\n"
            
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", summary)

        except Exception as e:
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", f"CRITICAL ERROR: {str(e)}\n\nCheck logs for details.")

    def export_csv(self):
        if not self.results:
            self.textbox.insert("end", "\n\nError: No analysis to export. Please run analysis first.")
            return

        filepath = fd.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Inventory as CSV"
        )
        
        if filepath:
            try:
                with open(filepath, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Produce Item", "Amount", "Analysis Timestamp"])
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for item, count in self.results["inventory"].items():
                        if count > 0:
                            writer.writerow([item, count, timestamp])
                
                self.textbox.insert("end", f"\n\n--> Successfully exported inventory to '{os.path.basename(filepath)}'")
            except Exception as e:
                self.textbox.insert("end", f"\n\nError exporting file to {str(e)}")

# Driver
if __name__ == "__main__":
    app = PantryApp()
    app.mainloop()
