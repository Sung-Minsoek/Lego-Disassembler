# Lego-Disassembler
We detect Lego parts from a Lego structure using deep-learning technique.

## Requirements
This is develop enviroment

- python v3.x
- torch 2.1.0

## Usage
### 1. Download Dataset zip file [here](https://drive.google.com/file/d/1WK9Bdq1EsHVLvtmXEcsbfY363Qg2ZUQo/view?usp=drive_link).
### 2. Unzip to data directory.
### 3. Execute main.py. It will do training model and test each epoch, finally show result graph.
#### *** If you want, use colab file in colab directory and you can also see result directly. ***

## Overview
The goal of this project is the deep-learning model captures the number of Lego bricks in an Lego structure.
> - Use 4-views of an Lego structure images as input, the model show the number of each type of brick(Regression problem).
> - We use 7 types of brick(1x1 brick, 1x2 brick, 1x3 brick, 1x4 brick, 2x2 brick, 2x3 brick, 2x4 brick).
> - For the model prediction performance, use another model(called Lego-GPT) trained by easier problem than out goal about Lego.

## Approach
> Simply, We think just give images to model and train repeatly, but it's not good. So we need other approaches for more prediction performance. From [GPT](https://arxiv.org/abs/2005.14165), we can get an idea. GPT is transformer model trained by the method that fill in the blank of a sentence in wikipedia or other. Through this way, the model can train relationships between each words and use the knowledge to any other language problems. We use this train method too. To solve our problem, the model learns "space information" of each type of brick and "connection rules" between bricks. We need to define new problem to be able to learn that knowledges and easier than origin problem. We called this problem and model is Lego-GPT. In short, the model learn space information and connection rules of Lego bricks solving Lego-GPT, then fine-tuning this model to our goal.

## Problems
### 1. Regression
- Use 4-views of an Lego structure images as input, the model show the number of each type of brick.
- Data is a tensor-4D(4, C, H, W).

### 2. Lego-GPT
- Input is 2 images. First is origin, second is one brick dettached from origin.
- Use these images, the model answer the dettached brick type.
- Data is a tensor-4D(2, C, H, W).

## Dataset
We have two types of dataset. One is for Lego-GPT. The other is for Regression problem. Each structure has 5~10 bricks in Regression dataset, 5~6 bricks in Lego-GPT dataset.

### 1. Regression Dataset
- Use 4-views of an Lego structure images as input, the model show the number of each type of brick.
- Label means the number of bricks(1: 2x4, 2: 2x3, 3: 2x2, 4: 1x2, 5: 1x1, 6: 1x4, 7: 1x3).

<img src="https://github.com/Sung-Minsoek/Lego-Disassembler/assets/127949889/8295efbf-b87d-40e7-92e2-8f1e8d67bd49" width="600" height="140">
<img src="https://github.com/Sung-Minsoek/Lego-Disassembler/assets/127949889/80b9e77f-4f81-41af-a9e7-b546d76aed08" width="600" height="140">

### 2. Lego-GPT Dataset
- Input is 2 images. First is origin, second is one brick dettached from origin.
- Label means dettached brick type(1: 2x4, 2: 2x3, 3: 2x2, 4: 1x2, 5: 1x1, 6: 1x4, 7: 1x3).

<img src="https://github.com/Sung-Minsoek/Lego-Disassembler/assets/127949889/320c79a6-b21f-4af9-945a-591510a99281" width="300" height="140">
<img src="https://github.com/Sung-Minsoek/Lego-Disassembler/assets/127949889/fed74b7c-2076-41c7-afb0-639e6d5c91df" width="300" height="140"><br/>
<img src="https://github.com/Sung-Minsoek/Lego-Disassembler/assets/127949889/3ca68f20-3dd6-45b5-9600-aa2b1fc695a1" width="300" height="140">
<img src="https://github.com/Sung-Minsoek/Lego-Disassembler/assets/127949889/af292569-8364-4e5e-b287-8fe38fa1e80c" width="300" height="140">


