Geometry.Tolerance = 1e-09;
// Define points
p0 = newp; Point(p0) = {2.8421067937708395, 9.745784368258573, 2.3637235361854936, 0.4939360978439978 };
p1 = newp; Point(p1) = {0.9479852941436946, 8.88129430871218, 3.534063410406182, 0.48843215191813244 };
p2 = newp; Point(p2) = {1.3742822765057419, 9.119413721435208, 3.522401714953546, 0.48843215191813244 };
p3 = newp; Point(p3) = {7.315489592735841, 12.438031622898498, 3.35987522940595, 0.9689309812508438 };
p4 = newp; Point(p4) = {5.592785430707364, 11.082825001955985, 1.1358392655664604, 0.48737113880568506 };
p5 = newp; Point(p5) = {5.107481104762237, 10.779716760340776, 0.9639939588007549, 0.5974325939999573 };
p6 = newp; Point(p6) = {2.5745088157502574, 10.134429067251446, 2.2177115885326044, 0.4939360978439978 };
p7 = newp; Point(p7) = {1.0869386500225857, 9.56977258583903, 3.4410362762094855, 0.5403793121961685 };
p8 = newp; Point(p8) = {7.547721099067371, 13.376440049383834, 3.425338188367685, 0.9689309812508438 };
p9 = newp; Point(p9) = {5.453632672504382, 11.504829940299846, 0.9356419408124892, 0.11492043833968017 };
p10 = newp; Point(p10) = {5.389513578395077, 11.44752293110289, 0.8594096929400413, 0.11492043833968017 };
p11 = newp; Point(p11) = {4.8422585922864965, 10.995228536778566, 0.3527949943271372, 0.7002517269389195 };
p12 = newp; Point(p12) = {-2.0, 5.0, 7.5, 6.0 };
p13 = newp; Point(p13) = {-2.0, 5.0, -2.5, 6.0 };
p14 = newp; Point(p14) = {-2.0, 15.0, -2.5, 6.0 };
p15 = newp; Point(p15) = {-2.0, 15.0, 7.5, 6.0 };
p16 = newp; Point(p16) = {8.0, 5.0, 7.5, 6.0 };
p17 = newp; Point(p17) = {8.0, 5.0, -2.5, 6.0 };
p18 = newp; Point(p18) = {8.0, 15.0, -2.5, 5.523027779961944 };
p19 = newp; Point(p19) = {8.0, 15.0, 7.5, 4.409463912614359 };
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

