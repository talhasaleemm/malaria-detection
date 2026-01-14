from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'Malaria Object Detection System - Project Documentation', 0, 1, 'C')
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, title):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, title, 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, body):
        # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 10, body)
        # Line break
        self.ln()

    def add_code_block(self, code):
        self.set_font('Courier', '', 10)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 5, code, 1, 'L', True)
        self.ln()

def create_pdf(filename):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # SECTION 1: INTRODUCTION
    pdf.chapter_title('1. Project Overview')
    pdf.chapter_body(
        "This project implements a professional-grade Object Detection System for detecting malaria parasites "
        "in microscopy images. Unlike traditional binary classification (Infected vs Uninfected), this system "
        "identifies the exact location of parasites within a cell using the YOLOv11 architecture.\n\n"
        "Key Engineering Highlights:\n"
        "- Synthetic Data Pipeline: Generates object detection datasets from single-cell images.\n"
        "- State-of-the-Art Model: Uses YOLOv11 Nano for real-time inference.\n"
        "- MLOps: Deployed via Dockerized FastAPI backend and Streamlit frontend."
    )

    # SECTION 2: ARCHITECTURE & WORKFLOW
    pdf.chapter_title('2. System Architecture')
    pdf.chapter_body(
        "The system follows a microservice architecture:\n\n"
        "1. Data Layer: Fetches NIH data, synthesizes 'slides' with bounding box labels.\n"
        "2. Training Layer: Fine-tunes YOLOv11n on the synthetic dataset.\n"
        "3. Inference Layer (FastAPI): Exposes a REST API endpoint (/predict) that accepts images and returns JSON detections.\n"
        "4. Presentation Layer (Streamlit): A user-friendly web UI for uploading images and visualizing results."
    )

    # SECTION 3: CODE EXPLANATION
    pdf.chapter_title('3. Code Structure & Explanation')

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'A. Data Pipeline (data_pipeline.py)', 0, 1)
    pdf.chapter_body(
        "The 'CellCompositor' class is the core engine. It downloads the NIH dataset "
        "and randomly places single cells onto a blank canvas to create synthetic microscopy "
        "slides. Crucially, it calculates the bounding box coordinates (x, y, w, h) for every "
        "placed cell, allowing us to train an Object Detection model without manual annotation."
    )
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'B. Model Training (train_yolo.py)', 0, 1)
    pdf.chapter_body(
        "We utilize the 'ultralytics' library to train YOLOv11. The script points to the "
        "generated 'dataset.yaml' and runs for a specified number of epochs. We use the "
        "'yolo11n.pt' (Nano) weights for speed and efficiency."
    )

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'C. Backend API (main.py)', 0, 1)
    pdf.chapter_body(
        "Implemented using FastAPI. It has a single endpoint '/predict'. When an image is "
        "received, it is converted to a generic PIL format, passed through the loaded YOLO "
        "model, and the resulting bounding boxes are returned as a JSON response."
    )
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, 'D. Frontend UI (app.py)', 0, 1)
    pdf.chapter_body(
        "Built with Streamlit. It handles file uploads, sends the image to the FastAPI backend, "
        "and uses OpenCV to draw the returned bounding boxes onto the image for visualization. "
        "Red boxes indicate Parasitized cells, Green boxes indicate Uninfected."
    )

    # SECTION 4: HOW TO RUN
    pdf.chapter_title('4. How to Run')
    pdf.chapter_body(
        "1. Generate Data: python data_pipeline.py\n"
        "2. Train Model: python train_yolo.py\n"
        "3. Start Backend: uvicorn main:app --port 8000\n"
        "4. Start Frontend: streamlit run app.py"
    )

    pdf.output(filename, 'F')
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_pdf("Malaria_Detection_Project_Docs.pdf")
