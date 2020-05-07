Geometry.Tolerance = 1e-10;
// Define points
p0 = newp; Point(p0) = {28.421067937708465, 97.45784368258576, 23.63723536185489, 4.939360978439977 };
p1 = newp; Point(p1) = {9.479852941436945, 88.8129430871218, 35.34063410406182, 4.884321519181368 };
p2 = newp; Point(p2) = {13.742822765057458, 91.1941372143521, 35.224017149535456, 4.884321519181368 };
p3 = newp; Point(p3) = {73.15489592735841, 124.38031622898498, 33.5987522940595, 9.689309812508428 };
p4 = newp; Point(p4) = {55.927854307073645, 110.82825001955985, 11.358392655664604, 4.873711388056849 };
p5 = newp; Point(p5) = {51.07481104762237, 107.79716760340776, 9.639939588007548, 5.974325939999574 };
p6 = newp; Point(p6) = {25.745088157502636, 101.34429067251449, 22.177115885325996, 4.939360978439977 };
p7 = newp; Point(p7) = {10.869386500225858, 95.69772585839029, 34.410362762094856, 5.403793121961688 };
p8 = newp; Point(p8) = {75.47721099067371, 133.76440049383834, 34.25338188367685, 9.689309812508428 };
p9 = newp; Point(p9) = {54.536326725043814, 115.04829940299845, 9.3564194081249, 1.1492043833968055 };
p10 = newp; Point(p10) = {53.895135783950764, 114.4752293110289, 8.594096929400413, 1.1492043833968055 };
p11 = newp; Point(p11) = {48.42258592286497, 109.95228536778566, 3.5279499432713717, 7.002517269389194 };
p12 = newp; Point(p12) = {-20.0, 50.0, 75.0, 60.0 };
p13 = newp; Point(p13) = {-20.0, 50.0, -25.0, 60.0 };
p14 = newp; Point(p14) = {-20.0, 150.0, -25.0, 60.0 };
p15 = newp; Point(p15) = {-20.0, 150.0, 75.0, 60.0 };
p16 = newp; Point(p16) = {80.0, 50.0, 75.0, 60.0 };
p17 = newp; Point(p17) = {80.0, 50.0, -25.0, 60.0 };
p18 = newp; Point(p18) = {80.0, 150.0, -25.0, 55.23027779961946 };
p19 = newp; Point(p19) = {80.0, 150.0, 75.0, 44.09463912614359 };
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

