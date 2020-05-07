Geometry.Tolerance = 5.0000000000000005e-12;
// Define points
p0 = newp; Point(p0) = {828.2165091472817, 2067.7291643296976, 312.22246846093077, 194.59438425462056 };
p1 = newp; Point(p1) = {568.4213587541685, 1949.1568736517147, 472.7447072370983, 98.78721956879947 };
p2 = newp; Point(p2) = {189.5970588287389, 1776.2588617424358, 706.8126820812364, 97.68643038362742 };
p3 = newp; Point(p3) = {274.85645530114914, 1823.882744287042, 704.4803429907091, 97.68643038362742 };
p4 = newp; Point(p4) = {530.3425370546047, 1966.5912274880168, 697.4913165582731, 114.36751134134572 };
p5 = newp; Point(p5) = {1463.097918547168, 2487.6063245796995, 671.97504588119, 193.78619625016836 };
p6 = newp; Point(p6) = {1118.5570861414728, 2216.565000391197, 227.16785311329207, 100.14735824552422 };
p7 = newp; Point(p7) = {1021.4962209524473, 2155.943352068155, 192.79879176015095, 119.48651879999156 };
p8 = newp; Point(p8) = {514.901763150052, 2026.8858134502893, 443.5423177065206, 98.78721956879947 };
p9 = newp; Point(p9) = {217.3877300045173, 1913.9545171678055, 688.2072552418972, 108.07586243923348 };
p10 = newp; Point(p10) = {464.2651328664942, 2059.413706417875, 687.6074048078594, 114.36751134134572 };
p11 = newp; Point(p11) = {1509.5442198134742, 2675.2880098767664, 685.067637673537, 178.81306960174732 };
p12 = newp; Point(p12) = {1077.9027156790153, 2289.504586220578, 171.8819385880083, 100.14735824552422 };
p13 = newp; Point(p13) = {968.4517184572993, 2199.045707355713, 70.55899886542743, 90.58641112436925 };
p14 = newp; Point(p14) = {719.9990483031519, 2215.3233892318494, 246.10212115832528, 83.83180063059449 };
p15 = newp; Point(p15) = {367.8732056738052, 2158.8664242258565, 696.3938061112077, 138.77841170433587 };
p16 = newp; Point(p16) = {1504.504209642048, 2853.8337053975797, 676.69227088394, 178.81306960174732 };
p17 = newp; Point(p17) = {1038.064592641665, 2360.252740603255, 102.042023695851, 107.09804436981067 };
p18 = newp; Point(p18) = {923.8330467396358, 2261.015210486378, 21.82653846906355, 90.58641112436925 };
p19 = newp; Point(p19) = {771.3900991091324, 2223.5630081570316, 180.38424403198582, 83.83180063059449 };
p20 = newp; Point(p20) = {-400.0, 1000.0, 1500.0, 1200.0 };
p21 = newp; Point(p21) = {-400.0, 1000.0, -500.0, 1200.0 };
p22 = newp; Point(p22) = {-400.0, 3000.0, -500.0, 1200.0 };
p23 = newp; Point(p23) = {-400.0, 3000.0, 1500.0, 1200.0 };
p24 = newp; Point(p24) = {1600.0, 1000.0, 1500.0, 1200.0 };
p25 = newp; Point(p25) = {1600.0, 1000.0, -500.0, 1200.0 };
p26 = newp; Point(p26) = {1600.0, 3000.0, -500.0, 1042.833906355379 };
p27 = newp; Point(p27) = {1600.0, 3000.0, 1500.0, 841.6172814742667 };
// End of point specification

// Define lines 
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1 = newl; Line(frac_line_1) = {p0, p7};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2 = newl; Line(frac_line_2) = {p1, p2};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3 = newl; Line(frac_line_3) = {p2, p3};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4 = newl; Line(frac_line_4) = {p3, p4};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5 = newl; Line(frac_line_5) = {p4, p5};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6 = newl; Line(frac_line_6) = {p5, p6};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7 = newl; Line(frac_line_7) = {p6, p7};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8 = newl; Line(frac_line_8) = {p8, p9};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9 = newl; Line(frac_line_9) = {p8, p13};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10 = newl; Line(frac_line_10) = {p9, p10};
Physical Line("FRACTURE_TIP_10") = {frac_line_10};

frac_line_11 = newl; Line(frac_line_11) = {p10, p11};
Physical Line("FRACTURE_TIP_11") = {frac_line_11};

frac_line_12 = newl; Line(frac_line_12) = {p11, p12};
Physical Line("FRACTURE_TIP_12") = {frac_line_12};

frac_line_13 = newl; Line(frac_line_13) = {p12, p13};
Physical Line("FRACTURE_TIP_13") = {frac_line_13};

frac_line_14 = newl; Line(frac_line_14) = {p14, p15};
Physical Line("FRACTURE_TIP_14") = {frac_line_14};

frac_line_15 = newl; Line(frac_line_15) = {p14, p19};
Physical Line("FRACTURE_TIP_15") = {frac_line_15};

frac_line_16 = newl; Line(frac_line_16) = {p15, p16};
Physical Line("FRACTURE_TIP_16") = {frac_line_16};

frac_line_17 = newl; Line(frac_line_17) = {p16, p17};
Physical Line("FRACTURE_TIP_17") = {frac_line_17};

frac_line_18 = newl; Line(frac_line_18) = {p17, p18};
Physical Line("FRACTURE_TIP_18") = {frac_line_18};

frac_line_19 = newl; Line(frac_line_19) = {p18, p19};
Physical Line("FRACTURE_TIP_19") = {frac_line_19};

frac_line_20 = newl; Line(frac_line_20) = {p20, p21};
Physical Line("DOMAIN_BOUNDARY_20") = {frac_line_20};

frac_line_21 = newl; Line(frac_line_21) = {p20, p23};
Physical Line("DOMAIN_BOUNDARY_21") = {frac_line_21};

frac_line_22 = newl; Line(frac_line_22) = {p20, p24};
Physical Line("DOMAIN_BOUNDARY_22") = {frac_line_22};

frac_line_23 = newl; Line(frac_line_23) = {p21, p22};
Physical Line("DOMAIN_BOUNDARY_23") = {frac_line_23};

frac_line_24 = newl; Line(frac_line_24) = {p21, p25};
Physical Line("DOMAIN_BOUNDARY_24") = {frac_line_24};

frac_line_25 = newl; Line(frac_line_25) = {p22, p23};
Physical Line("DOMAIN_BOUNDARY_25") = {frac_line_25};

frac_line_26 = newl; Line(frac_line_26) = {p22, p26};
Physical Line("DOMAIN_BOUNDARY_26") = {frac_line_26};

frac_line_27 = newl; Line(frac_line_27) = {p23, p27};
Physical Line("DOMAIN_BOUNDARY_27") = {frac_line_27};

frac_line_28 = newl; Line(frac_line_28) = {p24, p25};
Physical Line("DOMAIN_BOUNDARY_28") = {frac_line_28};

frac_line_29 = newl; Line(frac_line_29) = {p24, p27};
Physical Line("DOMAIN_BOUNDARY_29") = {frac_line_29};

frac_line_30 = newl; Line(frac_line_30) = {p25, p26};
Physical Line("DOMAIN_BOUNDARY_30") = {frac_line_30};

frac_line_31 = newl; Line(frac_line_31) = {p26, p27};
Physical Line("DOMAIN_BOUNDARY_31") = {frac_line_31};

// End of line specification 

// Start domain specification
// Start boundary surface specification
frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_20, frac_line_23, frac_line_25, -frac_line_21};
boundary_surface_3 = news; Plane Surface(boundary_surface_3) = {frac_loop_3};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_3") = {boundary_surface_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_28, frac_line_30, frac_line_31, -frac_line_29};
boundary_surface_4 = news; Plane Surface(boundary_surface_4) = {frac_loop_4};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_4") = {boundary_surface_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_20, frac_line_24, -frac_line_28, -frac_line_22};
boundary_surface_5 = news; Plane Surface(boundary_surface_5) = {frac_loop_5};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_5") = {boundary_surface_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_25, frac_line_27, -frac_line_31, -frac_line_26};
boundary_surface_6 = news; Plane Surface(boundary_surface_6) = {frac_loop_6};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_6") = {boundary_surface_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_23, frac_line_26, -frac_line_30, -frac_line_24};
boundary_surface_7 = news; Plane Surface(boundary_surface_7) = {frac_loop_7};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_7") = {boundary_surface_7};

frac_loop_8 = newll; 
Line Loop(frac_loop_8) = { frac_line_21, frac_line_27, -frac_line_29, -frac_line_22};
boundary_surface_8 = news; Plane Surface(boundary_surface_8) = {frac_loop_8};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_8") = {boundary_surface_8};

domain_loop = newsl;
Surface Loop(domain_loop) = { boundary_surface_3, boundary_surface_4, boundary_surface_5, boundary_surface_6, boundary_surface_7, boundary_surface_8};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_2, frac_line_3, frac_line_4, frac_line_5, frac_line_6, frac_line_7, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};


frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_8, frac_line_10, frac_line_11, frac_line_12, frac_line_13, -frac_line_9};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};


frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_14, frac_line_16, frac_line_17, frac_line_18, frac_line_19, -frac_line_15};
fracture_2 = news; Plane Surface(fracture_2) = {frac_loop_2};
Physical Surface("FRACTURE_2") = {fracture_2};
Surface{fracture_2} In Volume{1};


// End of fracture specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p20};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p21};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p22};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p23};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {p24};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {p25};
Physical Point("DOMAIN_BOUNDARY_POINT_6") = {p26};
Physical Point("DOMAIN_BOUNDARY_POINT_7") = {p27};
// End of physical point specification

