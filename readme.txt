Start by copying the repository to your local computer.
Open the project folder in a code editor, for example VS Code.
Make sure that you have Python installed.
Create a virtual environment by running 'python -m venv .venv' in your Terminal.
Then activate the virtual environment by running '.venv/bin/activate' on Windows or 'source .venv/bin/activate' on Mac OS.
Once your virtual environment is activated, run 'pip install -r requirements.txt'.
This should install all the dependencies listed in the requirements.txt file. You need these to run the code.

The code is divided into different folders: data, figures, helpers.
And there's a main_workflow.py Python file for the main task.
In the main workflow file, first, there are some global variables defined -> imported from config,
that will later affect the flow of data cleaning and transformation,
as the helper functions are being called and used.

The main model to train is a RandomForestClassifier with optimized hyperparameters.
And at the end, the model is once more evaluated for its accuracy.

The .env file that defines certain global variables is not provided publicly.
