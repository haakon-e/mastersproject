Geometry.Tolerance = 6.666666666666667e-12;
// Define points
p0 = newp; Point(p0) = {-500.0, 0.0, 1000.0, 150.0 };
p1 = newp; Point(p1) = {-500.0, 0.0, -500.0, 150.0 };
p2 = newp; Point(p2) = {-500.0, 1500.0, -500.0, 150.0 };
p3 = newp; Point(p3) = {-500.0, 1500.0, 1000.0, 150.0 };
p4 = newp; Point(p4) = {1000.0, 0.0, 1000.0, 150.0 };
p5 = newp; Point(p5) = {1000.0, 0.0, -500.0, 150.0 };
p6 = newp; Point(p6) = {1000.0, 1500.0, -500.0, 150.0 };
p7 = newp; Point(p7) = {1000.0, 1500.0, 1000.0, 150.0 };
// End of point specification

// Define lines 
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Physical Line("DOMAIN_BOUNDARY_0") = {frac_line_0};

frac_line_1 = newl; Line(frac_line_1) = {p0, p3};
Physical Line("DOMAIN_BOUNDARY_1") = {frac_line_1};

frac_line_2 = newl; Line(frac_line_2) = {p0, p4};
Physical Line("DOMAIN_BOUNDARY_2") = {frac_line_2};

frac_line_3 = newl; Line(frac_line_3) = {p1, p2};
Physical Line("DOMAIN_BOUNDARY_3") = {frac_line_3};

frac_line_4 = newl; Line(frac_line_4) = {p1, p5};
Physical Line("DOMAIN_BOUNDARY_4") = {frac_line_4};

frac_line_5 = newl; Line(frac_line_5) = {p2, p3};
Physical Line("DOMAIN_BOUNDARY_5") = {frac_line_5};

frac_line_6 = newl; Line(frac_line_6) = {p2, p6};
Physical Line("DOMAIN_BOUNDARY_6") = {frac_line_6};

frac_line_7 = newl; Line(frac_line_7) = {p3, p7};
Physical Line("DOMAIN_BOUNDARY_7") = {frac_line_7};

frac_line_8 = newl; Line(frac_line_8) = {p4, p5};
Physical Line("DOMAIN_BOUNDARY_8") = {frac_line_8};

frac_line_9 = newl; Line(frac_line_9) = {p4, p7};
Physical Line("DOMAIN_BOUNDARY_9") = {frac_line_9};

frac_line_10 = newl; Line(frac_line_10) = {p5, p6};
Physical Line("DOMAIN_BOUNDARY_10") = {frac_line_10};

frac_line_11 = newl; Line(frac_line_11) = {p6, p7};
Physical Line("DOMAIN_BOUNDARY_11") = {frac_line_11};

// End of line specification 

// Start domain specification
// Start boundary surface specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_3, frac_line_5, -frac_line_1};
boundary_surface_0 = news; Plane Surface(boundary_surface_0) = {frac_loop_0};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_0") = {boundary_surface_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_8, frac_line_10, frac_line_11, -frac_line_9};
boundary_surface_1 = news; Plane Surface(boundary_surface_1) = {frac_loop_1};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_1") = {boundary_surface_1};

frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_0, frac_line_4, -frac_line_8, -frac_line_2};
boundary_surface_2 = news; Plane Surface(boundary_surface_2) = {frac_loop_2};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_2") = {boundary_surface_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_5, frac_line_7, -frac_line_11, -frac_line_6};
boundary_surface_3 = news; Plane Surface(boundary_surface_3) = {frac_loop_3};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_3") = {boundary_surface_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_3, frac_line_6, -frac_line_10, -frac_line_4};
boundary_surface_4 = news; Plane Surface(boundary_surface_4) = {frac_loop_4};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_4") = {boundary_surface_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_1, frac_line_7, -frac_line_9, -frac_line_2};
boundary_surface_5 = news; Plane Surface(boundary_surface_5) = {frac_loop_5};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_5") = {boundary_surface_5};

domain_loop = newsl;
Surface Loop(domain_loop) = { boundary_surface_0, boundary_surface_1, boundary_surface_2, boundary_surface_3, boundary_surface_4, boundary_surface_5};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
// End of fracture specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p0};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p1};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p2};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p3};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {p4};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {p5};
Physical Point("DOMAIN_BOUNDARY_POINT_6") = {p6};
Physical Point("DOMAIN_BOUNDARY_POINT_7") = {p7};
// End of physical point specification

