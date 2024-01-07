# IN3063 Mathematics and Programming for AI Group 10
This is a guide on how to prepare and run the files for both tasks.

## Task 1
To open and run the task 1 file **NNfinal.py**:
  1. Use any Python IDE, Spyder is recommended but Jupyter Notebook can also be used.
  2. The libraries ```matplotlib``` and ```numpy``` is already configured within Spyder and Jupyter Notebook so you wont need to download external libraries.
  3. ```tensorflow``` and ```keras``` would need to be installed if Spyder is used but not  Jupyter Notebook.

Here are the steps to install ```tensorflow``` and ```keras``` through the terminal in the console of Spyder:
  1. After you have Spyder environment running, put the following commands into the Spyder terminal:
      ```
        -conda install tensorflow
        -conda install keras
      ```
  2. If you’re having trouble getting ```tensorflow``` to work try:
      ```
      -pip install --ignore-installed 
      --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl
      ```
  3. If you want the fancy GPU version of ```tensorflow```, just type:
      ```
      -pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-win_amd64.whl
      ```

## Task 2
To open and run the task 2 file **task2.py**:
  1. Use any Python IDE, Spyder is recommended but Jupyter Notebook can also be used.
  2. You would need to download ```Pytorch``` on Spyder terminal and Jupyter Notebook, here is a guide on downloading it: https://pytorch.org/get-started/locally/
  3. As an alternative, you can install ```Pytorch``` by running this command in the terminal:
      ```
      -pip install torch torchvision torchaudio
      ```
  4. Download the **dataset.zip** file and extract the file **brewery_data_complete_extended.csv**. You must save this file in the same folder as the file **task2.py**.
  5. You can now run **task2.py**.
