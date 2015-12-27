// Creating a mesh for the unit circle
 // Maximal width in the mesh h0=0.1
 h0= 0.1;
 radius = 1.0;
// Creating the points
Point(1) = {0, 0, 0, h0};
Point(2) = {-radius, 0, 0, h0};
Point(3) = {0, radius, 0, h0};
Point(4) = {radius, 0, 0, h0};
Point(5) = {0, -radius, 0, h0};
Circle(6) = {2, 1, 3};
Circle(7) = {3, 1, 4};
Circle(8) = {4, 1, 5};
Circle(9) = {5, 1, 2};
// Define a surface by a Line Loop
Line Loop(10) = {6, 7, 8, 9};
Plane Surface(11) = {10};
// Define a surface by a Line Loop
Physical Line(101) = {6, 7, 8, 9};
Physical Surface(201) = {11};