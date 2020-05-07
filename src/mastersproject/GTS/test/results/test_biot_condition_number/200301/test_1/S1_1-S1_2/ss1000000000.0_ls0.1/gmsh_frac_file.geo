Geometry.Tolerance = 1.0000000000000001e-11;
// Define points
p0 = newp; Point(p0) = {284.2106793770847, 974.5784368258576, 236.37235361854886, 49.393609784399715 };
p1 = newp; Point(p1) = {94.79852941436945, 888.1294308712179, 353.4063410406182, 48.84321519181371 };
p2 = newp; Point(p2) = {137.42822765057457, 911.941372143521, 352.24017149535456, 48.84321519181371 };
p3 = newp; Point(p3) = {731.548959273584, 1243.8031622898498, 335.987522940595, 96.89309812508418 };
p4 = newp; Point(p4) = {559.2785430707364, 1108.2825001955985, 113.58392655664603, 48.737113880568444 };
p5 = newp; Point(p5) = {510.74811047622364, 1077.9716760340775, 96.39939588007547, 59.74325939999578 };
p6 = newp; Point(p6) = {257.4508815750264, 1013.4429067251448, 221.7711588532599, 49.393609784399715 };
p7 = newp; Point(p7) = {108.69386500225858, 956.9772585839028, 344.10362762094854, 54.03793121961678 };
p8 = newp; Point(p8) = {754.7721099067371, 1337.6440049383832, 342.5338188367685, 96.89309812508418 };
p9 = newp; Point(p9) = {545.3632672504381, 1150.4829940299844, 93.5641940812489, 11.492043833967898 };
p10 = newp; Point(p10) = {538.9513578395076, 1144.752293110289, 85.94096929400412, 11.492043833967898 };
p11 = newp; Point(p11) = {484.22585922864965, 1099.5228536778566, 35.27949943271371, 70.02517269389193 };
p12 = newp; Point(p12) = {-200.0, 500.0, 750.0, 600.0 };
p13 = newp; Point(p13) = {-200.0, 500.0, -250.0, 600.0 };
p14 = newp; Point(p14) = {-200.0, 1500.0, -250.0, 600.0 };
p15 = newp; Point(p15) = {-200.0, 1500.0, 750.0, 600.0 };
p16 = newp; Point(p16) = {800.0, 500.0, 750.0, 600.0 };
p17 = newp; Point(p17) = {800.0, 500.0, -250.0, 600.0 };
p18 = newp; Point(p18) = {800.0, 1500.0, -250.0, 552.3027779961945 };
p19 = newp; Point(p19) = {800.0, 1500.0, 750.0, 440.946391261436 };
// End of point specification

// Define lines 
frac_line_0= newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1= newl; Line(frac_line_1) = {p0, p5};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2= newl; Line(frac_line_2) = {p1, p2};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3= newl; Line(frac_line_3) = {p2, p3};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4= newl; Line(frac_line_4) = {p3, p4};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5= newl; Line(frac_line_5) = {p4, p5};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6= newl; Line(frac_line_6) = {p6, p7};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7= newl; Line(frac_line_7) = {p6, p11};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8= newl; Line(frac_line_8) = {p7, p8};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9= newl; Line(frac_line_9) = {p8, p9};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10= newl; Line(frac_line_10) = {p9, p10};
Physical Line("FRACTURE_TIP_10") = {frac_line_10};

frac_line_11= newl; Line(frac_line_11) = {p10, p11};
Physical Line("FRACTURE_TIP_11") = {frac_line_11};

frac_line_12= newl; Line(frac_line_12) = {p12, p13};
Physical Line("AUXILIARY_LINE_12") = {frac_line_12};

frac_line_13= newl; Line(frac_line_13) = {p12, p15};
Physical Line("AUXILIARY_LINE_13") = {frac_line_13};

frac_line_14= newl; Line(frac_line_14) = {p12, p16};
Physical Line("AUXILIARY_LINE_14") = {frac_line_14};

frac_line_15= newl; Line(frac_line_15) = {p13, p14};
Physical Line("AUXILIARY_LINE_15") = {frac_line_15};

frac_line_16= newl; Line(frac_line_16) = {p13, p17};
Physical Line("AUXILIARY_LINE_16") = {frac_line_16};

frac_line_17= newl; Line(frac_line_17) = {p14, p15};
Physical Line("AUXILIARY_LINE_17") = {frac_line_17};

frac_line_18= newl; Line(frac_line_18) = {p14, p18};
Physical Line("AUXILIARY_LINE_18") = {frac_line_18};

frac_line_19= newl; Line(frac_line_19) = {p15, p19};
Physical Line("AUXILIARY_LINE_19") = {frac_line_19};

frac_line_20= newl; Line(frac_line_20) = {p16, p17};
Physical Line("AUXILIARY_LINE_20") = {frac_line_20};

frac_line_21= newl; Line(frac_line_21) = {p16, p19};
Physical Line("AUXILIARY_LINE_21") = {frac_line_21};

frac_line_22= newl; Line(frac_line_22) = {p17, p18};
Physical Line("AUXILIARY_LINE_22") = {frac_line_22};

frac_line_23= newl; Line(frac_line_23) = {p18, p19};
Physical Line("AUXILIARY_LINE_23") = {frac_line_23};

// End of line specification 

// Start domain specification
frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_12, frac_line_15, frac_line_17, -frac_line_13};
auxiliary_2 = news; Plane Surface(auxiliary_2) = {frac_loop_2};
Physical Surface("AUXILIARY_2") = {auxiliary_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_20, frac_line_22, frac_line_23, -frac_line_21};
auxiliary_3 = news; Plane Surface(auxiliary_3) = {frac_loop_3};
Physical Surface("AUXILIARY_3") = {auxiliary_3};

frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_12, frac_line_16, -frac_line_20, -frac_line_14};
auxiliary_4 = news; Plane Surface(auxiliary_4) = {frac_loop_4};
Physical Surface("AUXILIARY_4") = {auxiliary_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_17, frac_line_19, -frac_line_23, -frac_line_18};
auxiliary_5 = news; Plane Surface(auxiliary_5) = {frac_loop_5};
Physical Surface("AUXILIARY_5") = {auxiliary_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_15, frac_line_18, -frac_line_22, -frac_line_16};
auxiliary_6 = news; Plane Surface(auxiliary_6) = {frac_loop_6};
Physical Surface("AUXILIARY_6") = {auxiliary_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_13, frac_line_19, -frac_line_21, -frac_line_14};
auxiliary_7 = news; Plane Surface(auxiliary_7) = {frac_loop_7};
Physical Surface("AUXILIARY_7") = {auxiliary_7};

domain_loop = newsl;
Surface Loop(domain_loop) = {auxiliary_2,auxiliary_3,auxiliary_4,auxiliary_5,auxiliary_6,auxiliary_7};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_2, frac_line_3, frac_line_4, frac_line_5, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};


frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_6, frac_line_8, frac_line_9, frac_line_10, frac_line_11, -frac_line_7};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};


// End of fracture specification

// Start physical point specification
// End of physical point specification

