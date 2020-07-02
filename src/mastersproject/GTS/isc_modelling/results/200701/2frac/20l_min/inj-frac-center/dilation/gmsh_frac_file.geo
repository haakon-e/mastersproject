Geometry.Tolerance = 4.266666666666667e-10;
// Define points
p0 = newp; Point(p0) = {2.5549930708483672, 8.123886115035639, 1.2855021670281181, 0.46599717305184046 };
p1 = newp; Point(p1) = {0.8491708203301457, 7.47638483268674, 2.688309590788661, 0.46875 };
p2 = newp; Point(p2) = {5.896657108646382, 10.450343788581119, 2.676045459662254, 0.46875 };
p3 = newp; Point(p3) = {4.210557483121153, 8.943377289924133, 0.6714138226094074, 0.46875 };
p4 = newp; Point(p4) = {3.783014525223825, 8.590022294358254, 0.27562108931807605, 0.46875 };
p5 = newp; Point(p5) = {3.8907717571287335, 7.821747449313251, 0.5403016992922268, 0.46875 };
p6 = newp; Point(p6) = {5.626532988317818, 8.284628883936298, 2.5786428298648114, 0.46875 };
p7 = newp; Point(p7) = {4.357089774399731, 8.513949435768664, 2.770528094386547, 0.46875 };
p8 = newp; Point(p8) = {1.7467974917909093, 8.807617705803576, 2.6380297860355895, 0.46875 };
p9 = newp; Point(p9) = {-7.8125, 0.0, 15.625, 1.40625 };
p10 = newp; Point(p10) = {-7.8125, 0.0, -7.8125, 1.40625 };
p11 = newp; Point(p11) = {-7.8125, 23.4375, -7.8125, 1.40625 };
p12 = newp; Point(p12) = {-7.8125, 23.4375, 15.625, 1.40625 };
p13 = newp; Point(p13) = {15.625, 0.0, 15.625, 1.40625 };
p14 = newp; Point(p14) = {15.625, 0.0, -7.8125, 1.40625 };
p15 = newp; Point(p15) = {15.625, 23.4375, -7.8125, 1.40625 };
p16 = newp; Point(p16) = {15.625, 23.4375, 15.625, 1.40625 };
p17 = newp; Point(p17) = {2.8842763237498525, 8.67546094550248, 2.6833647926605737, 0.46875 };
p18 = newp; Point(p18) = {2.765724334694709, 8.339081453200674, 1.641081505800869, 0.46599717305184046 };
// End of point specification

// Define lines 
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1 = newl; Line(frac_line_1) = {p0, p4};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2 = newl; Line(frac_line_2) = {p1, p17};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3 = newl; Line(frac_line_3) = {p2, p3};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4 = newl; Line(frac_line_4) = {p2, p17};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5 = newl; Line(frac_line_5) = {p3, p4};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6 = newl; Line(frac_line_6) = {p5, p6};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7 = newl; Line(frac_line_7) = {p5, p18};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8 = newl; Line(frac_line_8) = {p6, p7};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9 = newl; Line(frac_line_9) = {p7, p8};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10 = newl; Line(frac_line_10) = {p8, p18};
Physical Line("FRACTURE_TIP_10") = {frac_line_10};

frac_line_11 = newl; Line(frac_line_11) = {p9, p10};
Physical Line("DOMAIN_BOUNDARY_11") = {frac_line_11};

frac_line_12 = newl; Line(frac_line_12) = {p9, p12};
Physical Line("DOMAIN_BOUNDARY_12") = {frac_line_12};

frac_line_13 = newl; Line(frac_line_13) = {p9, p13};
Physical Line("DOMAIN_BOUNDARY_13") = {frac_line_13};

frac_line_14 = newl; Line(frac_line_14) = {p10, p11};
Physical Line("DOMAIN_BOUNDARY_14") = {frac_line_14};

frac_line_15 = newl; Line(frac_line_15) = {p10, p14};
Physical Line("DOMAIN_BOUNDARY_15") = {frac_line_15};

frac_line_16 = newl; Line(frac_line_16) = {p11, p12};
Physical Line("DOMAIN_BOUNDARY_16") = {frac_line_16};

frac_line_17 = newl; Line(frac_line_17) = {p11, p15};
Physical Line("DOMAIN_BOUNDARY_17") = {frac_line_17};

frac_line_18 = newl; Line(frac_line_18) = {p12, p16};
Physical Line("DOMAIN_BOUNDARY_18") = {frac_line_18};

frac_line_19 = newl; Line(frac_line_19) = {p13, p14};
Physical Line("DOMAIN_BOUNDARY_19") = {frac_line_19};

frac_line_20 = newl; Line(frac_line_20) = {p13, p16};
Physical Line("DOMAIN_BOUNDARY_20") = {frac_line_20};

frac_line_21 = newl; Line(frac_line_21) = {p14, p15};
Physical Line("DOMAIN_BOUNDARY_21") = {frac_line_21};

frac_line_22 = newl; Line(frac_line_22) = {p15, p16};
Physical Line("DOMAIN_BOUNDARY_22") = {frac_line_22};

frac_line_23 = newl; Line(frac_line_23) = {p17, p18};
Physical Line("FRACTURE_LINE_23") = {frac_line_23};

// End of line specification 

// Start domain specification
// Start boundary surface specification
frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_11, frac_line_14, frac_line_16, -frac_line_12};
boundary_surface_2 = news; Plane Surface(boundary_surface_2) = {frac_loop_2};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_2") = {boundary_surface_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_19, frac_line_21, frac_line_22, -frac_line_20};
boundary_surface_3 = news; Plane Surface(boundary_surface_3) = {frac_loop_3};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_3") = {boundary_surface_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_11, frac_line_15, -frac_line_19, -frac_line_13};
boundary_surface_4 = news; Plane Surface(boundary_surface_4) = {frac_loop_4};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_4") = {boundary_surface_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_16, frac_line_18, -frac_line_22, -frac_line_17};
boundary_surface_5 = news; Plane Surface(boundary_surface_5) = {frac_loop_5};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_5") = {boundary_surface_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_14, frac_line_17, -frac_line_21, -frac_line_15};
boundary_surface_6 = news; Plane Surface(boundary_surface_6) = {frac_loop_6};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_6") = {boundary_surface_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_12, frac_line_18, -frac_line_20, -frac_line_13};
boundary_surface_7 = news; Plane Surface(boundary_surface_7) = {frac_loop_7};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_7") = {boundary_surface_7};

domain_loop = newsl;
Surface Loop(domain_loop) = { boundary_surface_2, boundary_surface_3, boundary_surface_4, boundary_surface_5, boundary_surface_6, boundary_surface_7};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_2, -frac_line_4, frac_line_3, frac_line_5, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};

Line{frac_line_23} In Surface{fracture_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_6, frac_line_8, frac_line_9, frac_line_10, -frac_line_7};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};

Line{frac_line_23} In Surface{fracture_1};

// End of fracture specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p9};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p10};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p11};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p12};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {p13};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {p14};
Physical Point("DOMAIN_BOUNDARY_POINT_6") = {p15};
Physical Point("DOMAIN_BOUNDARY_POINT_7") = {p16};
// End of physical point specification

