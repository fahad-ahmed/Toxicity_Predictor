Reproducibility:
Github Link: https://github.com/fahad-ahmed/Toxicity_Predictor

Step1:
Clone the code in your pc using git bash command:
Git clone https://github.com/fahad-ahmed/Toxicity_Predictor.git

Step2:
Download the dataset(Rat.tar.gz File of 550 MB) from the following
link: https://zenodo.org/record/3359047#.Y5kz33bMK3C

Step 3:
Extract Rat.tar.gz, you will find another file named ‘Rat.tar’. Extract also this file to find the
folder named “Rat”. Copy the folder and paste it into cloned source directory

Note: You can avoid step 4 to Step 6 if you open the main.py file in any python IDE

Step 4:
Open Command Prompt/Terminal and go to the source code directory.
Step 5:
Make sure “pandas” , ‘scikit-learn’ and ‘matplotlib’ are installed. Command
pip install -U pandas scikit-learn matplotlib
Step 6:
Run the main.py file using the following command.
python main.py


Step 7:
Observe the result in the command prompt. Confusion matrix plot files image will be generated
inside the dataset directories for various classifiers.

Note: The whole program loads 4 datasets and runs 3 machine-learning models for each of the
datasets. So, the whole program execution might take 4-5 minutes.