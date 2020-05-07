Geometry.Tolerance = 1.0000000000000001e-11;
// Define points
p0 = newp; Point(p0) = {94.79852941436945, 888.1294308712179, 353.4063410406182, 100.0 };
p1 = newp; Point(p1) = {731.548959273584, 1243.8031622898498, 335.987522940595, 100.0 };
p2 = newp; Point(p2) = {559.2785430707364, 1108.2825001955985, 113.58392655664603, 59.74325939999578 };
p3 = newp; Point(p3) = {510.74811047622364, 1077.9716760340775, 96.39939588007547, 59.74325939999578 };
p4 = newp; Point(p4) = {498.0187849124779, 1001.1836735120961, 69.15861750940508, 82.46510211515923 };
p5 = newp; Point(p5) = {720.1962225046807, 1060.4324971438461, 330.06628222269586, 100.0 };
p6 = newp; Point(p6) = {557.7074911231656, 1089.785527778389, 354.62759608147803, 89.38549007792243 };
p7 = newp; Point(p7) = {223.59007894923639, 1127.3750663428577, 337.66781261255545, 100.0 };
p8 = newp; Point(p8) = {-200.0, 500.0, 750.0, 600.0 };
p9 = newp; Point(p9) = {-200.0, 500.0, -250.0, 600.0 };
p10 = newp; Point(p10) = {-200.0, 1500.0, -250.0, 600.0 };
p11 = newp; Point(p11) = {-200.0, 1500.0, 750.0, 600.0 };
p12 = newp; Point(p12) = {800.0, 500.0, 750.0, 600.0 };
p13 = newp; Point(p13) = {800.0, 500.0, -250.0, 600.0 };
p14 = newp; Point(p14) = {800.0, 1500.0, -250.0, 586.1592711281697 };
p15 = newp; Point(p15) = {800.0, 1500.0, 750.0, 491.6591256043916 };
p16 = newp; Point(p16) = {469.3853701459429, 1097.3647813880173, 343.159217936503, 89.38549007792243 };
p17 = newp; Point(p17) = {423.59950920869153, 1038.1964559576547, 150.24677740551076, 100.0 };
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

