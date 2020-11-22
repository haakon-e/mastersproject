Geometry.Tolerance = 2.5e-09;
// Define points
p0 = newp; Point(p0) = {-1.0, 0.0, -1.0, 1.0 };
p1 = newp; Point(p1) = {1.0, 0.0, -1.0, 1.0 };
p2 = newp; Point(p2) = {1.0, 0.0, 1.0, 1.0 };
p3 = newp; Point(p3) = {-1.0, 0.0, 1.0, 1.0 };
// End of point specification

// Define lines 
frac_line_0= newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1= newl; Line(frac_line_1) = {p0, p3};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2= newl; Line(frac_line_2) = {p1, p2};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3= newl; Line(frac_line_3) = {p2, p3};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

// End of line specification 

// Start domain specification
domain_loop = newsl;
Surface Loop(domain_loop) = };
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_2, frac_line_3, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};


// End of fracture specification

// Start physical point specification
// End of physical point specification

