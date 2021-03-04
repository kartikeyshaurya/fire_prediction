# Forest Fire Prediction 

## Motivation
<p>
Forest Fires cause severe hazard in endangering vegetation, Animal and human life across the world, apart from this it can be a factor for Airborne hazards , Water pollution and other Post- fire risks. Also after Post-fire  is very hard to control and even wildland fire fighters face several life-threatening hazards including heat stress, fatigue, smoke and dust, as well as the risk of other injuries such as burns, cuts and scrapes, animal bites, and even rhabdomyolysis. Between 2000–2016, more than 350 wildland firefighters died on-duty. Only Amazon forest fire cost Brazil US$957 billion to US$3.5 trillion over a 30-year period.
</p>
<br>
<br>

# CNN Based Early stage detection classifier 
<br>

<img src="readme_files/work.gif">


 The working model is deployed [here](https://share.streamlit.io/kartikeyshaurya/fire_prediction/app.py)


<br>


# File Structure
<br>

```
FIRE_PREDICTION
├───data
│   ├───fire
│   └───Non_fire
├───model
│   ├───fire_detection.model
│   │   └───variables
│   └───sample_classifier_model  ## classic model 
├───readme_files  ## extra files for the readme.md 
├───techniques    ## source code 

```

## How to setup 

<p> follow these simple steps to setup the environment</p>

```
Step 1: make sure you have python installed version 3.5 to 3.7 (anyone) 
Step 2 : make a virtual environment( from my prefrence i use miniconda )
Step 3 : use this command to automatically download each required package :
                    pip install -r requirements.txt 

if everything didnot go as planned use this command 
                    pip install -r requirements_old.txt

step 4 : open the directory and run the app by :
                    streamlit run app.py 

step 5 : everything should be in order, and tryout my app :
    
```

## Tools used 
<img src="readme_files/A.png">

##  Thanks 
@sabstian thrun <br>
@ udacity </br>
@ Lovely Professional University <br>

## license 

<p> the license is MIT but it would be better if you include me(karikeyshaurya) also </p>