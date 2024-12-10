# MNIST Classification

## Target:
- The target of this project is to build a model that can classify MNIST digits with an accuracy of 99.4% consistently.
- The model should be able to achieve this accuracy with less than 10,000 parameters and under 15 epochs.
- A systematic approach is followed to build the model and achieve the target accuracy.
- Train the final model on AWS EC2 instance



## Approach:

- The model is built in 3 steps
- For each step, certain target is set and after achieving the target, results are noted
- Based on the results, analysis is done and target for the next step is planned



## Directory Structure:

- `model_1.py` contains the step 1 model
- `model_2.py` contains the step 2 model
- `model_3.py` contains the step 3 model
- `step_1.ipynb` is the notebook to train the step 1 model
- `step_2.ipynb` is the notebook to train the step 2 model
- `step_3.ipynb` is the notebook to train the step 3 model
- `RF Calculations.xlsx` contains the Receptive Field calculations done for models
- `requirements.txt` contains the dependencies for the project



## Steps:

### Step 1: [[Setup and Data Understanding]](./step_1.ipynb)
```markdown
### Target:
- Get the setup correct
- Understand the dataset
- Basic data transformations
- Setup the model skeleton with proper receptive field calculations


### Results:
- Parameters: 10,554
- Best Train Accuracy: 98.38%
- Best Test Accuracy: 99.09% (13th epoch)


### Analysis:
- The setup and basic transformations worked well
- The model skeleton seems good since it is able to achieve 99.09% accuracy under 15 epochs and 10.5K parameters
- Model accuracy seems saturating, need to add batch normalization and dropout layers
- User Global Average Pooling instead of normal convolutional layer to reduce the number of parameters
```


### Step 2: [[Regularization]](./step_2.ipynb)
```markdown
### Target:
- Add batch normalization to the model
- Add dropout to the model
- Use Global Average Pooling instead of normal convolutional layer

### Results:
- Parameters: 7,498
- Best Train Accuracy: 98.47%
- Best Test Accuracy: 99.30% (13th epoch)

### Analysis:
- Test accuracy increased from 99.09% to 99.30% in 13th epoch
- The model seems to be saturated on the training dataset
- Need to add more augmentation techniques to improve the model accuracy
- Need to use learning rate scheduler to achieve accuracy goals quicker
```

### Step 3: [[Auggmentation and Learning Rate]](./step_3.ipynb)
```markdown
### Target:
- Use cutout augmentation technique to improve the model accuracy
- Use learning rate scheduler to achieve accuracy goals quicker

### Results:
- Parameters: 9,776
- Best Train Accuracy: 97.32%
- Best Test Accuracy: 99.47% (14th epoch)

### Analysis:
- After adding cutout augmentation, the model accuracy has improved
- Model accuracy got saturated around 99.35%, so increased the capacity of the model by increaseing the number of channels in the convolutional layers
- Model was able to achieve the target accuracy of 99.4+% consistently from 10th epoch
```