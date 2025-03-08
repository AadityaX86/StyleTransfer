# Style Transfer

Implementation of Our Bachelor's in Computer Engineering Minor Project Paper.<br>
[Arbitrary Style Transfer for Nepali Landscapes and Sites using Transformers](https://drive.google.com/file/d/1R8rbPvl3D_ewrIzncPU__DkSjzLUnwjs/view?usp=drive_link)


# Team Members
[Aaditya Joshi](https://github.com/AadityaX86)<br>
[Abhijeet K.C.](https://github.com/Abhijeet-KC)<br> 
[Ankit Neupane](https://github.com/AnkitNeupane007)<br>
[Lijan Shrestha](https://github.com/Lijan09)

# Usage
## Inference/Testing on Local Machine
### 1. Clone the Repository  
```bash
git clone https://github.com/AadityaX86/StyleTransfer.git
cd StyleTransfer
```

### 2. Set Up a Virtual Environment  
It is recommended to use a virtual environment to manage dependencies.  

- **For Windows**  
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **For macOS/Linux**  
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

### 3. Install Requirements  
Ensure you have Python installed (Python 3.11.* preferred).  
Then, install dependencies from `requirements.txt`:  
```bash
pip install -r requirements.txt
```
***Make sure you have the [Checkpoint](#more-info) downloaded if you are not using your Trained Model***

### 4. Prepare Input Images  
Place your content and style images in the `./Evaluation` directory and rename them as follows:  
- **Content Image** → `image_content.*`  
- **Style Image** → `image_style.*`  
- Preferred formats: `.jpg` or `.png`  

### 5. Run the Code  
Execute the following command in your terminal:  
```bash
python main.py
```

## Training on Local Machine
- Make a Directory `.\Data\train\content` and `.\Data\train\style` and put respective Content Images and Style Images there.
- Make a Directory `.\.models\models_scratch` and `.\.models\models_checkpoint` and put the Checkpoint Model there.
- Make a Directory `.\.logs\logs_scratch` and `.\.logs\logs_checkpoint`
- Run the Following Python Command in Your Terminal for Training from Scratch
```
python .\train_scratch.py
```
- Run the Following Python Command in Your Terminal for Training from Checkpoint
```
python .\train_checkpoint.py
```


### More Info

- You can Download the Checkpoint At: [Checkpoint](https://drive.google.com/drive/folders/1UO77oZv8S5HGPnhdRRZnYzc43GS0R7WO?usp=drive_link)
- You can also Check Out the Repository to Run Through the Website At: https://github.com/Abhijeet-KC/StyleTransferFrontEnd