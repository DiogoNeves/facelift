# facelift
A simple Python tool to create animations from photos of a person, face aligned, in the order they were taken.

This is very simple and experimental, the goal is to learn something.  
It requires all photos to have the same person, alone.

## Brief

Create a tool that given a list of photos of the same person, aligns their faces and creates a GIF animation of the photos, sorted by the date they were taken.

### Strategy

1. Detect face in all photos and keep the Face Rect information for all photos
2. Find the central point where to centre all faces minimising total amount of moving
3. Find average face size so to minimise total resizing
4. Find the maximum rectangle that can show all centred photos within their boundaries (confusing point...)
5. Cut all photos
6. Sort them by date
7. Create a GIF

## Future

Depending on how much fun I have with this, I will add support for selecting (automatically or manually) the right face in photos of multiple people.
