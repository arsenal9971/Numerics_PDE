// Creating a square mesh from scratch

// Maximal width in the mesh h0=sqrt(2)/35

h0=0.040406101782088436;

// Xmax and Ymax of the rectangular domain

Xmax=1.0;

Ymax=1.0;

// We creates the points that define the domain

Point(1)={0,0,0,h0};

Point(2)={Xmax,0,0,h0};

Point(3)={Xmax,Ymax,0,h0};

Point(4)={0,Ymax,0,h0};

// Define the lines of the domian

Line(1)={1,2};
Line(2)={2,3};
Line(3)={3,4};
Line(4)={4,1};

// Define a surface by a Line Loop

Line Loop(5)={1,2,3,4};

// Defining finally a Surdace

Plane Surface(6) = {5} ;

Physical Line(101) = {1,2,3,4};
Physical Surface(201) = {6};
