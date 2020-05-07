Geometry.Tolerance = 1e-09;
// Define points
p0 = newp; Point(p0) = {0.9479852941436946, 8.88129430871218, 3.534063410406182, 1.0 };
p1 = newp; Point(p1) = {7.315489592735841, 12.438031622898498, 3.35987522940595, 1.0 };
p2 = newp; Point(p2) = {5.592785430707364, 11.082825001955985, 1.1358392655664604, 0.5974325939999573 };
p3 = newp; Point(p3) = {5.107481104762237, 10.779716760340776, 0.9639939588007549, 0.5974325939999573 };
p4 = newp; Point(p4) = {4.980187849124779, 10.011836735120962, 0.6915861750940507, 0.824651021151592 };
p5 = newp; Point(p5) = {7.201962225046808, 10.604324971438462, 3.300662822226959, 1.0 };
p6 = newp; Point(p6) = {5.577074911231656, 10.897855277783892, 3.5462759608147807, 0.8938549007792223 };
p7 = newp; Point(p7) = {2.235900789492364, 11.273750663428578, 3.3766781261255545, 1.0 };
p8 = newp; Point(p8) = {-2.0, 5.0, 7.5, 6.0 };
p9 = newp; Point(p9) = {-2.0, 5.0, -2.5, 6.0 };
p10 = newp; Point(p10) = {-2.0, 15.0, -2.5, 6.0 };
p11 = newp; Point(p11) = {-2.0, 15.0, 7.5, 6.0 };
p12 = newp; Point(p12) = {8.0, 5.0, 7.5, 6.0 };
p13 = newp; Point(p13) = {8.0, 5.0, -2.5, 6.0 };
p14 = newp; Point(p14) = {8.0, 15.0, -2.5, 5.861592711281697 };
p15 = newp; Point(p15) = {8.0, 15.0, 7.5, 4.916591256043915 };
p16 = newp; Point(p16) = {4.693853701459431, 10.973647813880174, 3.43159217936503, 0.8938549007792223 };
p17 = newp; Point(p17) = {4.235995092086917, 10.381964559576549, 1.5024677740551073, 1.0 };
// End of point specification

// Define lines 
frac_line_0= newl; Line(frac_line_0) = {p0, p16};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1= newl; Line(frac_line_1) = {p0, p17};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2= newl; Line(frac_line_2) = {p1, p2};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3= newl; Line(frac_line_3) = {p1, p16};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4= newl; Line(frac_line_4) = {p2, p3};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5= newl; Line(frac_line_5) = {p3, p17};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6= newl; Line(frac_line_6) = {p4, p5};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7= newl; Line(frac_line_7) = {p4, p7};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8= newl; Line(frac_line_8) = {p5, p6};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9= newl; Line(frac_line_9) = {p6, p7};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10= newl; Line(frac_line_10) = {p8, p9};
Physical Line("AUXILIARY_LINE_10") = {frac_line_10};

frac_line_11= newl; Line(frac_line_11) = {p8, p11};
Physical Line("AUXILIARY_LINE_11") = {frac_line_11};

frac_line_12= newl; Line(frac_line_12) = {p8, p12};
Physical Line("AUXILIARY_LINE_12") = {frac_line_12};

frac_line_13= newl; Line(frac_line_13) = {p9, p10};
Physical Line("AUXILIARY_LINE_13") = {frac_line_13};

frac_line_14= newl; Line(frac_line_14) = {p9, p13};
Physical Line("AUXILIARY_LINE_14") = {frac_line_14};

frac_line_15= newl; Line(frac_line_15) = {p10, p11};
Physical Line("AUXILIARY_LINE_15") = {frac_line_15};

frac_line_16= newl; Line(frac_line_16) = {p10, p14};
Physical Line("AUXILIARY_LINE_16") = {frac_line_16};

frac_line_17= newl; Line(frac_line_17) = {p11, p15};
Physical Line("AUXILIARY_LINE_17") = {frac_line_17};

frac_line_18= newl; Line(frac_line_18) = {p12, p13};
Physical Line("AUXILIARY_LINE_18") = {frac_line_18};

frac_line_19= newl; Line(frac_line_19) = {p12, p15};
Physical Line("AUXILIARY_LINE_19") = {frac_line_19};

frac_line_20= newl; Line(frac_line_20) = {p13, p14};
Physical Line("AUXILIARY_LINE_20") = {frac_line_20};

frac_line_21= newl; Line(frac_line_21) = {p14, p15};
Physical Line("AUXILIARY_LINE_21") = {frac_line_21};

frac_line_22= newl; Line(frac_line_22) = {p16, p17};
Physical Line("FRACTURE_LINE_22") = {frac_line_22};

// End of line specification 

// Start domain specification
frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_10, frac_line_13, frac_line_15, -frac_line_11};
auxiliary_2 = news; Plane Surface(auxiliary_2) = {frac_loop_2};
Physical Surface("AUXILIARY_2") = {auxiliary_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_18, frac_line_20, frac_line_21, -frac_line_19};
auxiliary_3 = news; Plane Surface(auxiliary_3) = {frac_loop_3};
Physical Surface("AUXILIARY_3") = {auxiliary_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_10, frac_line_14, -frac_line_18, -frac_line_12};
auxiliary_4 = news; Plane Surface(auxiliary_4) = {frac_loop_4};
Physical Surface("AUXILIARY_4") = {auxiliary_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_15, frac_line_17, -frac_line_21, -frac_line_16};
auxiliary_5 = news; Plane Surface(auxiliary_5) = {frac_loop_5};
Physical Surface("AUXILIARY_5") = {auxiliary_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_13, frac_line_16, -frac_line_20, -frac_line_14};
auxiliary_6 = news; Plane Surface(auxiliary_6) = {frac_loop_6};
Physical Surface("AUXILIARY_6") = {auxiliary_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_11, frac_line_17, -frac_line_19, -frac_line_12};
auxiliary_7 = news; Plane Surface(auxiliary_7) = {frac_loop_7};
Physical Surface("AUXILIARY_7") = {auxiliary_7};

domain_loop = newsl;
Surface Loop(domain_loop) = {auxiliary_2,auxiliary_3,auxiliary_4,auxiliary_5,auxiliary_6,auxiliary_7};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, -frac_line_3, frac_line_2, frac_line_4, frac_line_5, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};

Line{frac_line_22} In Surface{fracture_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_6, frac_line_8, frac_line_9, -frac_line_7};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};

Line{frac_line_22} In Surface{fracture_1};

// End of fracture specification

// Start physical point specification
// End of physical point specification

