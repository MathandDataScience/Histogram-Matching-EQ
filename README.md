# Histogram-Matching-EQ
## Python code for Histogram matching or Histogram equalization

This code is an exercise in understanding histogram equalization and matching. scikit-image has built in tool for both operations but were not used. Instead the operations were coded from scratch.  

## Histogram matching 
Using the common photos of peppers_color.tif and cameraman.tif the histogram of the cameraman.tif was mapped to peppers_color.tif. Note that the size of these photos are the same.

<p align="center">
 <img  src="Pictures/cameraman hist.png" >
</p>
<p align="center">
 <em>Figure 1: cameraman histogram</em>
</p>

<p align="center">
 <img  src="Pictures/peppers hist.png" >
</p>
<p align="center">
 <em>Figure 2: peppers histogram</em>
</p>

### Histogram matching results

<p align="center">
 <img  src="Pictures/hist match.PNG" >
</p>
<p align="center">
 <em>Figure 3: peppers mapped to cameraman histogram </em>
</p>

note photos must be of the same size.

## Histogram equalization

Using the common photos of peppers_color.tif the histogram was equalized 


<p align="center">
 <img  src="Pictures/eq hist.png" >
</p>
<p align="center">
 <em>Figure 4: peppers eq histogram </em>
</p>

<p align="center">
 <img  src="Pictures/peppers hist eq.png" >
</p>
<p align="center">
 <em>Figure 5: peppers with equalized histogram </em>
</p>


see code for more detail
