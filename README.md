# AR visualization with ArUco markers 
## Introduction
Current e-book readers in the market like traditional paperbacks, do not provide chapter recap. Getting a chapter recap is vital to catching up with the current chapter for the readers. It allows readers to get reacquainted to the characters that are reintroduced into the scenes or some contextual knowledge or backstory for the readers to follow through the current series of events. In this paper, we propose using named entity recognition model and text-to-text LongT5 transformer trained with BOOKSUM dataset to implement a summarizer that is capable of providing a list of characters and locations that are present in the chapter and also a plot summary of chapter.

### Main functionalities
Produce a chapter recap with the following information
- List of characters that appears in the various scenes of each chapter
- List of locations of the scenes in each chapter
- A plot summary of each chapter

## Dependencies
### Input Data
- Alice’s Adventures in Wonderland by Lewis Carroll (obtained from NLTK Gutenberg Corpus).
### Training DataSet
- BOOKSUM
  * A collection of dataset for long-form narrative summarized jointly developed by Salesforce Research and Yale University.
  * Covers literature domain such as novels, plays and stories and human written summaries in three levels of details (paragraphs, chapter, book).
  * Readily available on the HuggingFace platform
  
### Software
- Natural Language Toolkit (NLTK)
- Flair Models
- HuggingFace

## Demo

* Overlay an image on top of the ArUco markers
![](https://github.com/francelow/ARvisualize/blob/main/visualize_img.gif)  
* Overlay a video on top of the ArUco markers 
![](https://github.com/francelow/ARvisualize/blob/main/visualize_video.gif)  
* Overlay a 3D model on top of the ArUco markers  
![](https://github.com/francelow/ARvisualize/blob/main/visualize_model.gif)  

## Authors
Francois Low
