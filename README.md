# facelift
A simple Python tool to create animations from photos of a person, face aligned, in the order they were taken.

This is a very simple and experimental tool. The goal is to learn something within a week or so.  
If you like the idea and want it, or parts of it, developed further, just send me a message.  

It requires all photos to have the same person, alone. For more information see README.md in the photos folder.

## Brief

Create a tool that given a list of photos of the same person, aligns their faces and creates a GIF animation of the photos, sorted by the date they were taken.

### Strategy

1. Detect face in all photos and keep the Face Rect information for all photos
2. Find average face size so to minimise total resizing 
3. Cut all photos by the minimum between a pre-defined square or intersection of all images
4. Sort them by date
5. Create a GIF

## Goals

* Learn a little more about Face Detection (using libraries)
* Practice Python outside what I currently do
* Have fun
* Create a timelapse of my daughter's face

## Future

Depending on how much fun I have with this, I will add support for selecting (automatically or manually) the right face in photos of multiple people.

## Run the Tool

Keep in mind this is very experimental, I didn't spend much time thinking about distribution.

### Installation

The hardest part will be installing opencv (you need to get to a point where this works):

```Python
import cv2
print dir(cv2)
```

I use (Brew)[http://brew.sh/] but still had to hack a couple of things.

After you get that working, create a new virtual environment and run:

```bash
> pip install -r requirements.txt
```

You should be ready to go.

### Tests

In the right environment:

```bash
> pytest test_facelift.py 
```

### Running

Add your photos to the `photos/` folder in the same directory as the tool.

```bash
> python facelift.py
```

This uses all the default values. For more information on options available
run `python facelift.py --help`.
