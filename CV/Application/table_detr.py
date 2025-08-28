## ref:https://medium.com/@shwet.prakash97/building-a-table-extractor-with-microsoft-table-transfomer-baf0c953a59d(main)
## ref:https://iamrajatroy.medium.com/document-intelligence-series-part-1-table-detection-with-yolo-1fa0a198fd7
## ref:https://iamrajatroy.medium.com/document-intelligence-series-part-2-transformer-for-table-detection-extraction-80a52486fa3
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download
from PIL import Image

import torch

from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection

import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

pytesseract = None
class TableExtractor:
    def __init__(self,
             detection_model_name: str = "microsoft/table-transformer-detection",
             structure_model_name: str = "microsoft/table-transformer-structure-recognition"):
        """
        Initializes the TableExtractor with specified models.
        """
        # if pytesseract is None:
        #     raise ImportError("Tesseract is not available. Please install it to use this class.")
        self.detection_model_name = detection_model_name
        self.structure_model_name = structure_model_name
        
        # Initialize models and feature extractor upon creation.
        self._initialize_models()

    def _initialize_models(self) -> None:
        """
        Private method to load and initialize the necessary models and feature extractor.
        """
        print("Initializing models...")
        # Determine the computation device (GPU if available, otherwise CPU).
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA (NVIDIA GPU).")
        elif torch.backends.mps.is_available(): # For Apple Silicon
            self.device = "mps"
            print("Using MPS (Apple Silicon GPU).")
        else:
            self.device = "cpu"
            print("Using CPU.")
        # Load the feature extractor, which preprocesses images for the models.
        self.feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
        # Load the pre-trained model for table detection.
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(
            self.detection_model_name
        ).to(self.device)
        
        # Load the pre-trained model for table structure recognition.
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            self.structure_model_name
        ).to(self.device)
        print("Models initialized successfully.")
    
    def extract_table(self, image_path: str, detection_flag:bool=True) -> pd.DataFrame:
        """
        The main public method to perform the full table extraction process.
        """
        image = Image.open(image_path).convert("RGB")
        if detection_flag:
            # Step 1: Detect the table's bounding box in the image.
            table_bounds = self._detect_table_bounds(image)
            if not table_bounds:
                print("No table detected in the image.")
                return pd.DataFrame() # Return an empty DataFrame if no table is found.
            
            # Crop the image to focus only on the first detected table.
            table_crop_img = image.crop(table_bounds[0])
            # Step 2: Recognize the structure (rows, columns) of the cropped table.
            structure_results = self._recognize_table_structure(table_crop_img)
            dataframe = self._build_dataframe_from_structure(table_crop_img, structure_results)

        else:
            structure_results = self._recognize_table_structure(image)
            # Step 3: Build a pandas DataFrame from the recognized structure and OCR.
            dataframe = self._build_dataframe_from_structure(image, structure_results)
        
        return dataframe
    
    def _detect_table_bounds(self, image: Image.Image) -> List[Tuple[float, float, float, float]]:
        """
        Detects table bounding boxes in the provided image.
        """
        # Prepare the image for the model.
        encoding = self.feature_extractor(image, return_tensors="pt")
        # Move input tensors to the same device as the model.
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Perform inference.
        with torch.no_grad():
            outputs = self.detection_model(**encoding)

        # Post-process the model's output to get scores, labels, and boxes.
        width, height = image.size
        results = self.feature_extractor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=[(height, width)]
        )[0]

        # Filter results to keep only those classified as 'table'.
        table_boxes = [box.tolist() for label, box in zip(results['labels'], results['boxes']) if self.detection_model.config.id2label[label.item()] == 'table']
        
        return table_boxes
    
    def _recognize_table_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        """
        Recognizes the structure (rows, columns, headers) of a cropped table image.
        """
        encoding = self.feature_extractor(table_image, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.structure_model(**encoding)

        width, height = table_image.size
        results = self.feature_extractor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=[(height, width)]
        )[0]

        return results
    
    def _get_cell_coordinates(self, table_structure: Dict[str, Any]) -> Tuple[List, List]:
        """
        Extracts and sorts the bounding boxes for rows and columns.
        """
        id2label = self.structure_model.config.id2label
        
        rows = [box.tolist() for label, box in zip(table_structure['labels'], table_structure['boxes']) if id2label[label.item()] == 'table row']
        columns = [box.tolist() for label, box in zip(table_structure['labels'], table_structure['boxes']) if id2label[label.item()] == 'table column']
        # Sort rows by their top coordinate (y-min) and columns by their left coordinate (x-min).
        rows.sort(key=lambda box: box[1])
        columns.sort(key=lambda box: box[0])
        
        return rows, columns
    
    def _get_intersection(self, box1: List[float], box2: List[float]) -> List[float]:
        """Calculates the intersection of two bounding boxes."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
                return []
        
        return [x_left, y_top, x_right, y_bottom]

    def _perform_ocr_on_cell(self, image: Image.Image, cell_box: List[float]) -> str:
        """Performs OCR on a specific cell region of an image."""
        cell_image = image.crop(cell_box)
        
        # Use pytesseract to extract text. --psm 6 assumes a single uniform block of text.
        text = ""#pytesseract.image_to_string(cell_image, config='--psm 6').strip()
        
        return text.replace("\n", " ")
    
    def _build_dataframe_from_structure(self, table_image: Image.Image, structure_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Constructs a pandas DataFrame by intersecting row and column boxes and performing OCR.
        """
        rows, columns = self._get_cell_coordinates(structure_results)
        data = [["" for _ in columns] for _ in rows]

        print(f"Building DataFrame from {len(rows)} rows and {len(columns)} columns...")
        for i, row_box in enumerate(rows):
            for j, col_box in enumerate(columns):
                # Find the intersection of the current row and column to define a cell.
                cell_box = self._get_intersection(row_box, col_box)
                
                if cell_box:
                    # Perform OCR on the cell's bounding box.
                    cell_text = self._perform_ocr_on_cell(table_image, cell_box)
                    data[i][j] = cell_text
        
        return pd.DataFrame(data)



if __name__ == "__main__":
    # ## model
    # feature_extractor = DetrFeatureExtractor()
    # model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    # file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_table.png")
    # image = Image.open(file_path).convert("RGB")

    # encoding = feature_extractor(image, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**encoding)

    # results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

    # labels, boxes = results['labels'], results['boxes']

    # column_header = None
    # table_rows = []
    # for label, (xmin, ymin, xmax, ymax) in zip(labels.tolist(), boxes.tolist()):
    #     label = label_dict[label]
    #     if label in ['table row', 'table column header']:
    #         cropped_image = image.crop((xmin, ymin, xmax, ymax))
    #         if label == "table column header":
    #             column_header = cropped_image
    #         else:
    #             table_rows.append(cropped_image)


    extractor = TableExtractor()
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_table.png")
    
    # Provide the path to your image
    extracted_df = extractor.extract_table(file_path)
    
    print("\n--- Extracted DataFrame ---")
    print(extracted_df)

    # Provide the path to your image
    extracted_df = extractor.extract_table(file_path,detection_flag=False)
    
    print("\n--- Extracted DataFrame ---")
    print(extracted_df)