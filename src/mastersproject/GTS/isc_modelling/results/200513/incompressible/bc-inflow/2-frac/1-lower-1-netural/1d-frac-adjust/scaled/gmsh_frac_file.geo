Geometry.Tolerance = 3.3333333333333335e-11;
// Define points
p0 = newp; Point(p0) = {249.99999999999997, 0.0, 300.0, 30.0 };
p1 = newp; Point(p1) = {250.00000000000003, 300.0, 300.0, 30.0 };
p2 = newp; Point(p2) = {50.00000000000003, 300.0, 0.0, 30.0 };
p3 = newp; Point(p3) = {49.999999999999964, 0.0, 0.0, 30.0 };
p4 = newp; Point(p4) = {300.0, 300.0, 50.0, 30.0 };
p5 = newp; Point(p5) = {300.0, 0.0, 50.0, 30.0 };
p6 = newp; Point(p6) = {50.0, 0.0, 300.0, 30.0 };
p7 = newp; Point(p7) = {50.0, 300.0, 300.0, 30.0 };
p8 = newp; Point(p8) = {0.0, 0.0, 300.0, 50.0 };
p9 = newp; Point(p9) = {0.0, 0.0, 0.0, 49.999999999999964 };
p10 = newp; Point(p10) = {0.0, 300.0, 0.0, 50.00000000000003 };
p11 = newp; Point(p11) = {0.0, 300.0, 300.0, 50.0 };
p12 = newp; Point(p12) = {300.0, 0.0, 300.0, 50.00000000000003 };
p13 = newp; Point(p13) = {300.0, 0.0, 0.0, 50.0 };
p14 = newp; Point(p14) = {300.0, 300.0, 0.0, 50.0 };
p15 = newp; Point(p15) = {300.0, 300.0, 300.0, 49.99999999999997 };
p16 = newp; Point(p16) = {169.99999999999997, 0.0, 180.00000000000003, 30.0 };
p17 = newp; Point(p17) = {170.00000000000003, 300.0, 179.99999999999997, 30.0 };
// End of point specification

// Define lines 
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_BOUNDARY_LINE_0") = {frac_line_0};

frac_line_1 = newl; Line(frac_line_1) = {p0, p6};
Physical Line("DOMAIN_BOUNDARY_1") = {frac_line_1};

frac_line_2 = newl; Line(frac_line_2) = {p0, p12};
Physical Line("DOMAIN_BOUNDARY_2") = {frac_line_2};

frac_line_3 = newl; Line(frac_line_3) = {p0, p16};
Physical Line("FRACTURE_BOUNDARY_LINE_3") = {frac_line_3};

frac_line_4 = newl; Line(frac_line_4) = {p1, p7};
Physical Line("DOMAIN_BOUNDARY_4") = {frac_line_4};

frac_line_5 = newl; Line(frac_line_5) = {p1, p15};
Physical Line("DOMAIN_BOUNDARY_5") = {frac_line_5};

frac_line_6 = newl; Line(frac_line_6) = {p1, p17};
Physical Line("FRACTURE_BOUNDARY_LINE_6") = {frac_line_6};

frac_line_7 = newl; Line(frac_line_7) = {p2, p3};
Physical Line("FRACTURE_BOUNDARY_LINE_7") = {frac_line_7};

frac_line_8 = newl; Line(frac_line_8) = {p2, p10};
Physical Line("DOMAIN_BOUNDARY_8") = {frac_line_8};

frac_line_9 = newl; Line(frac_line_9) = {p2, p14};
Physical Line("DOMAIN_BOUNDARY_9") = {frac_line_9};

frac_line_10 = newl; Line(frac_line_10) = {p2, p17};
Physical Line("FRACTURE_BOUNDARY_LINE_10") = {frac_line_10};

frac_line_11 = newl; Line(frac_line_11) = {p3, p9};
Physical Line("DOMAIN_BOUNDARY_11") = {frac_line_11};

frac_line_12 = newl; Line(frac_line_12) = {p3, p13};
Physical Line("DOMAIN_BOUNDARY_12") = {frac_line_12};

frac_line_13 = newl; Line(frac_line_13) = {p3, p16};
Physical Line("FRACTURE_BOUNDARY_LINE_13") = {frac_line_13};

frac_line_14 = newl; Line(frac_line_14) = {p4, p5};
Physical Line("FRACTURE_BOUNDARY_LINE_14") = {frac_line_14};

frac_line_15 = newl; Line(frac_line_15) = {p4, p14};
Physical Line("DOMAIN_BOUNDARY_15") = {frac_line_15};

frac_line_16 = newl; Line(frac_line_16) = {p4, p15};
Physical Line("DOMAIN_BOUNDARY_16") = {frac_line_16};

frac_line_17 = newl; Line(frac_line_17) = {p4, p17};
Physical Line("FRACTURE_BOUNDARY_LINE_17") = {frac_line_17};

frac_line_18 = newl; Line(frac_line_18) = {p5, p12};
Physical Line("DOMAIN_BOUNDARY_18") = {frac_line_18};

frac_line_19 = newl; Line(frac_line_19) = {p5, p13};
Physical Line("DOMAIN_BOUNDARY_19") = {frac_line_19};

frac_line_20 = newl; Line(frac_line_20) = {p5, p16};
Physical Line("FRACTURE_BOUNDARY_LINE_20") = {frac_line_20};

frac_line_21 = newl; Line(frac_line_21) = {p6, p7};
Physical Line("FRACTURE_BOUNDARY_LINE_21") = {frac_line_21};

frac_line_22 = newl; Line(frac_line_22) = {p6, p8};
Physical Line("DOMAIN_BOUNDARY_22") = {frac_line_22};

frac_line_23 = newl; Line(frac_line_23) = {p6, p16};
Physical Line("FRACTURE_BOUNDARY_LINE_23") = {frac_line_23};

frac_line_24 = newl; Line(frac_line_24) = {p7, p11};
Physical Line("DOMAIN_BOUNDARY_24") = {frac_line_24};

frac_line_25 = newl; Line(frac_line_25) = {p7, p17};
Physical Line("FRACTURE_BOUNDARY_LINE_25") = {frac_line_25};

frac_line_26 = newl; Line(frac_line_26) = {p8, p9};
Physical Line("DOMAIN_BOUNDARY_26") = {frac_line_26};

frac_line_27 = newl; Line(frac_line_27) = {p8, p11};
Physical Line("DOMAIN_BOUNDARY_27") = {frac_line_27};

frac_line_28 = newl; Line(frac_line_28) = {p9, p10};
Physical Line("DOMAIN_BOUNDARY_28") = {frac_line_28};

frac_line_29 = newl; Line(frac_line_29) = {p10, p11};
Physical Line("DOMAIN_BOUNDARY_29") = {frac_line_29};

frac_line_30 = newl; Line(frac_line_30) = {p12, p15};
Physical Line("DOMAIN_BOUNDARY_30") = {frac_line_30};

frac_line_31 = newl; Line(frac_line_31) = {p13, p14};
Physical Line("DOMAIN_BOUNDARY_31") = {frac_line_31};

frac_line_32 = newl; Line(frac_line_32) = {p16, p17};
Physical Line("FRACTURE_LINE_32") = {frac_line_32};

// End of line specification 

// Start domain specification
// Start boundary surface specification
frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_26, frac_line_28, frac_line_29, -frac_line_27};
boundary_surface_2 = news; Plane Surface(boundary_surface_2) = {frac_loop_2};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_2") = {boundary_surface_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_15, -frac_line_31, -frac_line_19, frac_line_18, frac_line_30, -frac_line_16};
boundary_surface_3 = news; Plane Surface(boundary_surface_3) = {frac_loop_3};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_3") = {boundary_surface_3};
Line{frac_line_14} In Surface{boundary_surface_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_1, frac_line_22, frac_line_26, -frac_line_11, frac_line_12, -frac_line_19, frac_line_18, -frac_line_2};
boundary_surface_4 = news; Plane Surface(boundary_surface_4) = {frac_loop_4};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_4") = {boundary_surface_4};
Line{frac_line_3} In Surface{boundary_surface_4};
Line{frac_line_13} In Surface{boundary_surface_4};
Line{frac_line_20} In Surface{boundary_surface_4};
Line{frac_line_23} In Surface{boundary_surface_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_4, frac_line_24, -frac_line_29, -frac_line_8, frac_line_9, -frac_line_15, frac_line_16, -frac_line_5};
boundary_surface_5 = news; Plane Surface(boundary_surface_5) = {frac_loop_5};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_5") = {boundary_surface_5};
Line{frac_line_6} In Surface{boundary_surface_5};
Line{frac_line_10} In Surface{boundary_surface_5};
Line{frac_line_17} In Surface{boundary_surface_5};
Line{frac_line_25} In Surface{boundary_surface_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_8, -frac_line_28, -frac_line_11, frac_line_12, frac_line_31, -frac_line_9};
boundary_surface_6 = news; Plane Surface(boundary_surface_6) = {frac_loop_6};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_6") = {boundary_surface_6};
Line{frac_line_7} In Surface{boundary_surface_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_1, frac_line_22, frac_line_27, -frac_line_24, -frac_line_4, frac_line_5, -frac_line_30, -frac_line_2};
boundary_surface_7 = news; Plane Surface(boundary_surface_7) = {frac_loop_7};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_7") = {boundary_surface_7};
Line{frac_line_0} In Surface{boundary_surface_7};
Line{frac_line_21} In Surface{boundary_surface_7};

domain_loop = newsl;
Surface Loop(domain_loop) = { boundary_surface_2, boundary_surface_3, boundary_surface_4, boundary_surface_5, boundary_surface_6, boundary_surface_7};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_6, -frac_line_10, frac_line_7, frac_line_13, -frac_line_3};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};

Line{frac_line_32} In Surface{fracture_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_14, frac_line_20, -frac_line_23, frac_line_21, frac_line_25, -frac_line_17};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};

Line{frac_line_32} In Surface{fracture_1};

// End of fracture specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p8};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p9};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p10};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p11};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {p12};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {p13};
Physical Point("DOMAIN_BOUNDARY_POINT_6") = {p14};
Physical Point("DOMAIN_BOUNDARY_POINT_7") = {p15};
Physical Point("FRACTURE_BOUNDARY_POINT_0") = {p0};
Physical Point("FRACTURE_BOUNDARY_POINT_1") = {p1};
Physical Point("FRACTURE_BOUNDARY_POINT_2") = {p2};
Physical Point("FRACTURE_BOUNDARY_POINT_3") = {p3};
Physical Point("FRACTURE_BOUNDARY_POINT_4") = {p4};
Physical Point("FRACTURE_BOUNDARY_POINT_5") = {p5};
Physical Point("FRACTURE_BOUNDARY_POINT_6") = {p6};
Physical Point("FRACTURE_BOUNDARY_POINT_7") = {p7};
Physical Point("FRACTURE_BOUNDARY_POINT_8") = {p16};
Physical Point("FRACTURE_BOUNDARY_POINT_9") = {p17};
// End of physical point specification

