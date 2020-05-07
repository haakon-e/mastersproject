Geometry.Tolerance = 1e-12;
// Define points
p0 = newp; Point(p0) = {2842.1067937708626, 9745.784368258583, 2363.723536185479, 493.93609784399825 };
p1 = newp; Point(p1) = {947.9852941436945, 8881.294308712178, 3534.063410406182, 488.4321519181327 };
p2 = newp; Point(p2) = {1374.2822765057422, 9119.413721435207, 3522.4017149535457, 488.4321519181327 };
p3 = newp; Point(p3) = {7315.489592735841, 12438.031622898498, 3359.87522940595, 968.9309812508425 };
p4 = newp; Point(p4) = {5592.785430707364, 11082.825001955984, 1135.8392655664604, 487.37113880568415 };
p5 = newp; Point(p5) = {5107.481104762237, 10779.716760340776, 963.9939588007549, 597.4325939999562 };
p6 = newp; Point(p6) = {2574.5088157502782, 10134.429067251454, 2217.7115885325875, 493.93609784399825 };
p7 = newp; Point(p7) = {1086.9386500225858, 9569.772585839028, 3441.0362762094855, 540.3793121961686 };
p8 = newp; Point(p8) = {7547.721099067371, 13376.440049383833, 3425.3381883676852, 968.9309812508425 };
p9 = newp; Point(p9) = {5453.632672504382, 11504.829940299844, 935.6419408124898, 114.92043833968032 };
p10 = newp; Point(p10) = {5389.513578395076, 11447.522931102889, 859.4096929400413, 114.92043833968032 };
p11 = newp; Point(p11) = {4842.258592286496, 10995.228536778566, 352.79499432713715, 700.2517269389194 };
p12 = newp; Point(p12) = {-2000.0, 5000.0, 7500.0, 6000.0 };
p13 = newp; Point(p13) = {-2000.0, 5000.0, -2500.0, 6000.0 };
p14 = newp; Point(p14) = {-2000.0, 15000.0, -2500.0, 6000.0 };
p15 = newp; Point(p15) = {-2000.0, 15000.0, 7500.0, 6000.0 };
p16 = newp; Point(p16) = {8000.0, 5000.0, 7500.0, 6000.0 };
p17 = newp; Point(p17) = {8000.0, 5000.0, -2500.0, 6000.0 };
p18 = newp; Point(p18) = {8000.0, 15000.0, -2500.0, 5523.027779961946 };
p19 = newp; Point(p19) = {8000.0, 15000.0, 7500.0, 4409.463912614359 };
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

