# README

This is a code for implementing lane detection in a video for an autonomous vehicle.
The code was part of the **"Complete Self-Driving Car Course - Applied Deep Learning"** course of Udemy. 

To make it work, we need to make use of ***cv2***, ***numpy***, and ***matplotlib.pyplot*** libraries.



## 1. Greyscale:
 
 First, we want to turn it to greyscale, so that we have to deal only with one channel (shades of black), instead of 3 (RGB).
We do that by using:

    `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)` 
    
<img width="960" alt="gray" src="https://user-images.githubusercontent.com/34197007/79237262-3fcb1e80-7e6e-11ea-8096-5a167dd98338.PNG">

## 2. Reduce Noise:

We run a kernel, size 5 by 5 of gaussian values to return another image that will be blurred to reduce the noise.
We do that by using:

    `cv2.GaussianBlur(grayImage, (5,5), 0)` 
    
<img width="960" alt="blur" src="https://user-images.githubusercontent.com/34197007/79237268-40fc4b80-7e6e-11ea-9eb0-77ac0fac6ee4.PNG">

## 3. Gradient image: 

We take the previous image, and we run derivative function along rows and columns. If the gradient in a grid is high enough - which means change of intensity of pixels and thus probably edge detection - then it colors it with white. Otherwise, if low change, it will be black.
We select a *high threshold = 150* and *low threshold = 50*. Usually ratio of high to low threshold equal to 3 is appropriate.
    `cv2.Canny(blurredImage, 50,150)` 

 - If gradient> high_threshold, then edge detected. 
 - If gradient < low_threshold, then no edge. 
 - If in between then it will be an edge only if it's next to a strong edge. 
 
<img width="960" alt="canny" src="https://user-images.githubusercontent.com/34197007/79237270-4194e200-7e6e-11ea-9615-75342d76915c.PNG">

## 4. Region of interest: 

We need to specify which is the region we are interested in applying edge detection. We do that using matplotlib to know which are the axis values to create the mask. We select a triangle mask:
    `cv2.fillPoly(mask, polygons, 255)` 
Where:
 - *mask* is an array the same shape as our image
 - *polygons*, is an array defining the shape in which we want to focus
 - and *255* is the color (white) of the mask.
 
<img width="960" alt="fillpoly" src="https://user-images.githubusercontent.com/34197007/79237261-3f328800-7e6e-11ea-9f91-0d654796f09c.PNG">

## 5. Bitwise AND:

We apply bitwise AND: This returns 1 only if at the same bit-positions of the compared bits there are 1s. Thus, applying it to the mask and the image, will result to 0 in the areas of 0 mask (black area), and 1 to the areas of 1 of the mask (white area, the area we focus) and 1 of the image (the area we detected an edge). Thaw way, we isolated the region of interest:

    `cv2.bitwise_and(image, mask)` 
    
<img width="960" alt="bitwise" src="https://user-images.githubusercontent.com/34197007/79237265-4063b500-7e6e-11ea-8a09-ef4bc6a9207f.PNG">

## 6. Hough Transformation:

We apply Hough Transformation to identify lines on our edges: 
We transform x and y in hough space to find which is the m and b (or ρ and θ) that fit our points as a straight line. We do that by:

*HoughLinesP(image,(resolution), threshold, placeHolder array, minLineLength, maxLineGap)*

`cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)` 

Where:
- cropped_image is the image after we applied region of interest.
- Resolution is 2 for ρ and 1 degree for θ. We can do trial and error here to find the best resolution.
- Threshold stands for how many lines must intersect in the same bin, to declare it as as a straight line. We define it as 100.
- placeHolder array is our initialization. 
- minLineLength is the min amount of pixels on the same line in order to define it a line. 
- maxLineGap is the max gap between lines in pixels, in order to combine them, and consider them as one line.

<img width="960" alt="houghLines" src="https://user-images.githubusercontent.com/34197007/79237263-3fcb1e80-7e6e-11ea-80d8-2a215a9655f7.PNG">

## 7. Average line:

Of all the lines we identified, we need to find the average of the lines, in order to combine some of them. We do that by using:

`np.polyfit`  to find the parameters b and m (y=mx+b) of each of the lines, and:
`np.average` to find their minimum.

We make use of `make_coordinates` function to find the (x1, y1) and (x2, y2) points of the lines.

<img width="960" alt="averagedLines" src="https://user-images.githubusercontent.com/34197007/79237264-4063b500-7e6e-11ea-8990-5929855f9048.PNG">

## 8. Display lines:

We make use of the `display_lines` functions to display the averaged lines we found, in our image. To do that, we use:
 `cv2.line(image, (x1,y1), (x2,y2), (255, 0, 0), 10)`

Where:
 - *image* is an array of the same shape as our image
 - *(x1, y1)* is the first point of the line
 - *(x2, y2)* is the second point of the line
 - *(255, 0, 0)* is the color of the line we want (blue)
 - And *10* the thickness. 

<img width="960" alt="display" src="https://user-images.githubusercontent.com/34197007/79237259-3e99f180-7e6e-11ea-9c03-a60c394631d5.PNG">

## 9. Blend the two images:

Sum (Blend) of the image containing the lines and original image.

 `cv2.addWeighted(image, 0.8, line, 1, 1)`

Where:

 - *image* is our original image.
 - *0.8* is the weight of the original image.
 - *line* is the image containing the lines.
 - *1* is the weight for line image.
 - *1* is a scalar value to be added to both the images.

<img width="960" alt="blend" src="https://user-images.githubusercontent.com/34197007/79237266-4063b500-7e6e-11ea-9942-c3ad000826a8.PNG">

## 10. Show the results:

 `cv2.imshow('result', combo_image)`
  `cv2.waitKey(0)`

Where:

 - *'result'* is the title we give and,
 - *combo_image*, is the image we combined both the original and the lines image.
 - With WaitKey, it displays the image for a specific amount of time. 0 means it will stay on display until we press a key.

Alternatively, we can do it with:
 `plt.imshow(image)`
  `plt.show()`

## 11. Video Capture:

To do the same but on a video frame instead of an image, we make use of:
 `cv2.VideoCapture("test.mp4")`
  
