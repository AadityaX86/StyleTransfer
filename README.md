# Style Transfer

Implementation of Our Unofficial Bachelor's in Computer Engineering Minor Project Paper.<br>
[Arbitrary Style Transfer for Nepali Landscapes and Sites using Transformers](https://drive.google.com/drive/folders/1FehV76RhBkTTr-fYzuRLq1Juu4DtJVnC?usp=sharing)


# Team Members
[Aaditya Joshi](https://github.com/AadityaX86)<br>
[Abhijeet K.C.](https://github.com/Abhijeet-KC)<br> 
[Ankit Neupane](https://github.com/AnkitNeupane007)<br>
[Lijan Shrestha](https://github.com/Lijan09)

# Usage
### Inference/Testing on Local Machine
- Clone the Repository<br>
```
git clone https://github.com/AadityaX86/StyleTransfer.git
```
- Install Requirements<br>
    - Install Python on Your Device (Python 3.11.*  Preferred)
    - Read the `requirements.txt` File Carefully and Install the Requirements.<br><br>
    ```
    pip install -r requirements.txt
    ```
- Put the Content Image & Style Image in the `.\Evaluation` Directory and Rename to:<br>
`.\image_content.*`<br>
`.\image_style.*`<br>
`.jpg` & `.png` Format are Preferred

- Run the Code
    - Run the Following Python Command in Your Terminal<br><br>
    ```
    python .\main.py
    ```
### Training on Local Machine
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
### Run Through Website

- You can also Check Out the Repository to Run Through the Website At: https://github.com/Abhijeet-KC/StyleTransferFrontEnd