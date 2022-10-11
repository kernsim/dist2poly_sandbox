# Hit test between line and polygons

Describe the polygon as 

- list of points
- pairs of indices of points describing the lines segement
- center point and bounding radius.

Describe the line as

- base point
- direction vector (normalized)
- normal vector (normalized)
- transformation, i.e. rotation matrix describing the direction and normal of the line.

Do the hit test:

1. calculate distance of center-point to the line and check if it smaller than the bounding radius,
2. transform all points of the polygon into the line's coordinate system, i.e. project all points onto the normal vector and line vector,
3. find the polygon's segments that cross x==0, i.e. the line,
4. calculate crossing coordinate.

Then collect all distances and find the smallest of the positive ones.

![example plot](/plot.png?raw=true "example plot")
