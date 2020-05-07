Geometry.Tolerance = 1e-07;
// Define points
p0 = newp; Point(p0) = {-0.02, 0.05, 0.075, 0.06 };
p1 = newp; Point(p1) = {-0.02, 0.05, -0.025, 0.06 };
p2 = newp; Point(p2) = {-0.02, 0.15, -0.025, 0.06 };
p3 = newp; Point(p3) = {-0.02, 0.15, 0.075, 0.06 };
p4 = newp; Point(p4) = {0.08, 0.05, 0.075, 0.06 };
p5 = newp; Point(p5) = {0.08, 0.05, -0.025, 0.06 };
p6 = newp; Point(p6) = {0.08, 0.15, -0.025, 0.06 };
p7 = newp; Point(p7) = {0.08, 0.15, 0.075, 0.06 };
// End of point specification

// Define lines 
frac_line_0= newl; Line(frac_line_0) = {p0, p1};
Physical Line("AUXILIARY_LINE_0") = {frac_line_0};

frac_line_1= newl; Line(frac_line_1) = {p0, p3};
Physical Line("AUXILIARY_LINE_1") = {frac_line_1};

frac_line_2= newl; Line(frac_line_2) = {p0, p4};
Physical Line("AUXILIARY_LINE_2") = {frac_line_2};

frac_line_3= newl; Line(frac_line_3) = {p1, p2};
Physical Line("AUXILIARY_LINE_3") = {frac_line_3};

frac_line_4= newl; Line(frac_line_4) = {p1, p5};
Physical Line("AUXILIARY_LINE_4") = {frac_line_4};

frac_line_5= newl; Line(frac_line_5) = {p2, p3};
Physical Line("AUXILIARY_LINE_5") = {frac_line_5};

frac_line_6= newl; Line(frac_line_6) = {p2, p6};
Physical Line("AUXILIARY_LINE_6") = {frac_line_6};

frac_line_7= newl; Line(frac_line_7) = {p3, p7};
Physical Line("AUXILIARY_LINE_7") = {frac_line_7};

frac_line_8= newl; Line(frac_line_8) = {p4, p5};
Physical Line("AUXILIARY_LINE_8") = {frac_line_8};

frac_line_9= newl; Line(frac_line_9) = {p4, p7};
Physical Line("AUXILIARY_LINE_9") = {frac_line_9};

frac_line_10= newl; Line(frac_line_10) = {p5, p6};
Physical Line("AUXILIARY_LINE_10") = {frac_line_10};

frac_line_11= newl; Line(frac_line_11) = {p6, p7};
Physical Line("AUXILIARY_LINE_11") = {frac_line_11};

// End of line specification 

// Start domain specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_3, frac_line_5, -frac_line_1};
auxiliary_0 = news; Plane Surface(auxiliary_0) = {frac_loop_0};
Physical Surface("AUXILIARY_0") = {auxiliary_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_8, frac_line_10, frac_line_11, -frac_line_9};
auxiliary_1 = news; Plane Surface(auxiliary_1) = {frac_loop_1};
Physical Surface("AUXILIARY_1") = {auxiliary_1};

frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_0, frac_line_4, -frac_line_8, -frac_line_2};
auxiliary_2 = news; Plane Surface(auxiliary_2) = {frac_loop_2};
Physical Surface("AUXILIARY_2") = {auxiliary_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_5, frac_line_7, -frac_line_11, -frac_line_6};
auxiliary_3 = news; Plane Surface(auxiliary_3) = {frac_loop_3};
Physical Surface("AUXILIARY_3") = {auxiliary_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_3, frac_line_6, -frac_line_10, -frac_line_4};
auxiliary_4 = news; Plane Surface(auxiliary_4) = {frac_loop_4};
Physical Surface("AUXILIARY_4") = {auxiliary_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_1, frac_line_7, -frac_line_9, -frac_line_2};
auxiliary_5 = news; Plane Surface(auxiliary_5) = {frac_loop_5};
Physical Surface("AUXILIARY_5") = {auxiliary_5};

domain_loop = newsl;
Surface Loop(domain_loop) = {auxiliary_0,auxiliary_1,auxiliary_2,auxiliary_3,auxiliary_4,auxiliary_5};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
// End of fracture specification

// Start physical point specification
// End of physical point specification

