Geometry.Tolerance = 5e-13;
// Define points
p0 = newp; Point(p0) = {5684.213587541725, 19491.568736517165, 4727.447072370958, 4000.0 };
p1 = newp; Point(p1) = {1895.970588287389, 17762.588617424357, 7068.126820812364, 4000.0 };
p2 = newp; Point(p2) = {14630.979185471682, 24876.063245796995, 6719.7504588119, 4000.0 };
p3 = newp; Point(p3) = {11185.570861414728, 22165.650003911967, 2271.678531132921, 4000.0 };
p4 = newp; Point(p4) = {10214.962209524474, 21559.433520681552, 1927.9879176015097, 4000.0 };
p5 = newp; Point(p5) = {5149.0176315005565, 20268.85813450291, 4435.423177065175, 4000.0 };
p6 = newp; Point(p6) = {2173.8773000451715, 19139.545171678055, 6882.072552418971, 4000.0 };
p7 = newp; Point(p7) = {15095.442198134742, 26752.880098767666, 6850.6763767353705, 4000.0 };
p8 = newp; Point(p8) = {10779.027156790153, 22895.045862205778, 1718.8193858800826, 4000.0 };
p9 = newp; Point(p9) = {9684.517184572993, 21990.457073557132, 705.5899886542743, 4000.0 };
p10 = newp; Point(p10) = {3678.7320567380525, 21588.66424225857, 6963.938061112077, 4000.0 };
p11 = newp; Point(p11) = {15045.042096420479, 28538.3370539758, 6766.922708839401, 4000.0 };
p12 = newp; Point(p12) = {10380.645926416651, 23602.527406032554, 1020.42023695851, 4000.0 };
p13 = newp; Point(p13) = {9238.33046739636, 22610.15210486378, 218.26538469063584, 4000.0 };
p14 = newp; Point(p14) = {7713.900991091325, 22235.630081570318, 1803.8424403198583, 4000.0 };
p15 = newp; Point(p15) = {9960.375698249558, 20023.673470241923, 1383.1723501881015, 4000.0 };
p16 = newp; Point(p16) = {14403.924450093615, 21208.649942876924, 6601.325644453918, 4000.0 };
p17 = newp; Point(p17) = {11154.14982246331, 21795.71055556778, 7092.551921629561, 4000.0 };
p18 = newp; Point(p18) = {4471.801578984728, 22547.501326857157, 6753.356252251109, 4000.0 };
p19 = newp; Point(p19) = {-4000.0, 10000.0, 15000.0, 4000.0 };
p20 = newp; Point(p20) = {-4000.0, 10000.0, -5000.0, 4000.0 };
p21 = newp; Point(p21) = {-4000.0, 30000.0, -5000.0, 4000.0 };
p22 = newp; Point(p22) = {-4000.0, 30000.0, 15000.0, 4000.0 };
p23 = newp; Point(p23) = {16000.0, 10000.0, 15000.0, 4000.0 };
p24 = newp; Point(p24) = {16000.0, 10000.0, -5000.0, 4000.0 };
p25 = newp; Point(p25) = {16000.0, 30000.0, -5000.0, 4000.0 };
p26 = newp; Point(p26) = {16000.0, 30000.0, 15000.0, 4000.0 };
p27 = newp; Point(p27) = {9387.707402918862, 21947.295627760348, 6863.18435873006, 4000.0 };
p28 = newp; Point(p28) = {8471.990184173832, 20763.929119153094, 3004.9355481102148, 4000.0 };
p29 = newp; Point(p29) = {7383.747388799624, 22209.180020486347, 6869.413869211068, 4000.0 };
p30 = newp; Point(p30) = {7080.254296818459, 21348.048520193723, 4201.16865485022, 4000.0 };
p31 = newp; Point(p31) = {5199.230039125803, 22465.66273012718, 6790.2804852411045, 4000.0 };
p32 = newp; Point(p32) = {5219.936760516049, 22203.484033725374, 6021.358574689523, 4000.0 };
// End of point specification

// Define lines 
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1 = newl; Line(frac_line_1) = {p0, p28};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2 = newl; Line(frac_line_2) = {p1, p27};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3 = newl; Line(frac_line_3) = {p2, p3};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4 = newl; Line(frac_line_4) = {p2, p27};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5 = newl; Line(frac_line_5) = {p3, p4};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6 = newl; Line(frac_line_6) = {p4, p28};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7 = newl; Line(frac_line_7) = {p5, p6};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8 = newl; Line(frac_line_8) = {p5, p9};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9 = newl; Line(frac_line_9) = {p6, p29};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10 = newl; Line(frac_line_10) = {p7, p8};
Physical Line("FRACTURE_TIP_10") = {frac_line_10};

frac_line_11 = newl; Line(frac_line_11) = {p7, p29};
Physical Line("FRACTURE_TIP_11") = {frac_line_11};

frac_line_12 = newl; Line(frac_line_12) = {p8, p9};
Physical Line("FRACTURE_TIP_12") = {frac_line_12};

frac_line_13 = newl; Line(frac_line_13) = {p10, p11};
Physical Line("FRACTURE_TIP_13") = {frac_line_13};

frac_line_14 = newl; Line(frac_line_14) = {p10, p14};
Physical Line("FRACTURE_TIP_14") = {frac_line_14};

frac_line_15 = newl; Line(frac_line_15) = {p11, p12};
Physical Line("FRACTURE_TIP_15") = {frac_line_15};

frac_line_16 = newl; Line(frac_line_16) = {p12, p13};
Physical Line("FRACTURE_TIP_16") = {frac_line_16};

frac_line_17 = newl; Line(frac_line_17) = {p13, p14};
Physical Line("FRACTURE_TIP_17") = {frac_line_17};

frac_line_18 = newl; Line(frac_line_18) = {p15, p16};
Physical Line("FRACTURE_TIP_18") = {frac_line_18};

frac_line_19 = newl; Line(frac_line_19) = {p15, p30};
Physical Line("FRACTURE_TIP_19") = {frac_line_19};

frac_line_20 = newl; Line(frac_line_20) = {p16, p17};
Physical Line("FRACTURE_TIP_20") = {frac_line_20};

frac_line_21 = newl; Line(frac_line_21) = {p17, p31};
Physical Line("FRACTURE_TIP_21") = {frac_line_21};

frac_line_22 = newl; Line(frac_line_22) = {p18, p31};
Physical Line("FRACTURE_TIP_22") = {frac_line_22};

frac_line_23 = newl; Line(frac_line_23) = {p18, p32};
Physical Line("FRACTURE_TIP_23") = {frac_line_23};

frac_line_24 = newl; Line(frac_line_24) = {p19, p20};
Physical Line("DOMAIN_BOUNDARY_24") = {frac_line_24};

frac_line_25 = newl; Line(frac_line_25) = {p19, p22};
Physical Line("DOMAIN_BOUNDARY_25") = {frac_line_25};

frac_line_26 = newl; Line(frac_line_26) = {p19, p23};
Physical Line("DOMAIN_BOUNDARY_26") = {frac_line_26};

frac_line_27 = newl; Line(frac_line_27) = {p20, p21};
Physical Line("DOMAIN_BOUNDARY_27") = {frac_line_27};

frac_line_28 = newl; Line(frac_line_28) = {p20, p24};
Physical Line("DOMAIN_BOUNDARY_28") = {frac_line_28};

frac_line_29 = newl; Line(frac_line_29) = {p21, p22};
Physical Line("DOMAIN_BOUNDARY_29") = {frac_line_29};

frac_line_30 = newl; Line(frac_line_30) = {p21, p25};
Physical Line("DOMAIN_BOUNDARY_30") = {frac_line_30};

frac_line_31 = newl; Line(frac_line_31) = {p22, p26};
Physical Line("DOMAIN_BOUNDARY_31") = {frac_line_31};

frac_line_32 = newl; Line(frac_line_32) = {p23, p24};
Physical Line("DOMAIN_BOUNDARY_32") = {frac_line_32};

frac_line_33 = newl; Line(frac_line_33) = {p23, p26};
Physical Line("DOMAIN_BOUNDARY_33") = {frac_line_33};

frac_line_34 = newl; Line(frac_line_34) = {p24, p25};
Physical Line("DOMAIN_BOUNDARY_34") = {frac_line_34};

frac_line_35 = newl; Line(frac_line_35) = {p25, p26};
Physical Line("DOMAIN_BOUNDARY_35") = {frac_line_35};

frac_line_36 = newl; Line(frac_line_36) = {p27, p28};
Physical Line("FRACTURE_LINE_36") = {frac_line_36};

frac_line_37 = newl; Line(frac_line_37) = {p29, p30};
Physical Line("FRACTURE_LINE_37") = {frac_line_37};

frac_line_38 = newl; Line(frac_line_38) = {p30, p32};
Physical Line("FRACTURE_TIP_38") = {frac_line_38};

frac_line_39 = newl; Line(frac_line_39) = {p31, p32};
Physical Line("FRACTURE_LINE_39") = {frac_line_39};

// End of line specification 

// Start domain specification
// Start boundary surface specification
frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_24, frac_line_27, frac_line_29, -frac_line_25};
boundary_surface_4 = news; Plane Surface(boundary_surface_4) = {frac_loop_4};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_4") = {boundary_surface_4};

frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_32, frac_line_34, frac_line_35, -frac_line_33};
boundary_surface_5 = news; Plane Surface(boundary_surface_5) = {frac_loop_5};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_5") = {boundary_surface_5};

frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_24, frac_line_28, -frac_line_32, -frac_line_26};
boundary_surface_6 = news; Plane Surface(boundary_surface_6) = {frac_loop_6};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_6") = {boundary_surface_6};

frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_29, frac_line_31, -frac_line_35, -frac_line_30};
boundary_surface_7 = news; Plane Surface(boundary_surface_7) = {frac_loop_7};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_7") = {boundary_surface_7};

frac_loop_8 = newll; 
Line Loop(frac_loop_8) = { frac_line_27, frac_line_30, -frac_line_34, -frac_line_28};
boundary_surface_8 = news; Plane Surface(boundary_surface_8) = {frac_loop_8};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_8") = {boundary_surface_8};

frac_loop_9 = newll; 
Line Loop(frac_loop_9) = { frac_line_25, frac_line_31, -frac_line_33, -frac_line_26};
boundary_surface_9 = news; Plane Surface(boundary_surface_9) = {frac_loop_9};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_9") = {boundary_surface_9};

domain_loop = newsl;
Surface Loop(domain_loop) = { boundary_surface_4, boundary_surface_5, boundary_surface_6, boundary_surface_7, boundary_surface_8, boundary_surface_9};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_2, -frac_line_4, frac_line_3, frac_line_5, frac_line_6, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};

Line{frac_line_36} In Surface{fracture_0};

frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_7, frac_line_9, -frac_line_11, frac_line_10, frac_line_12, -frac_line_8};
fracture_1 = news; Plane Surface(fracture_1) = {frac_loop_1};
Physical Surface("FRACTURE_1") = {fracture_1};
Surface{fracture_1} In Volume{1};

Line{frac_line_37} In Surface{fracture_1};

frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_13, frac_line_15, frac_line_16, frac_line_17, -frac_line_14};
fracture_2 = news; Plane Surface(fracture_2) = {frac_loop_2};
Physical Surface("FRACTURE_2") = {fracture_2};
Surface{fracture_2} In Volume{1};

Line{frac_line_39} In Surface{fracture_2};

frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_18, frac_line_20, frac_line_21, -frac_line_22, frac_line_23, -frac_line_38, -frac_line_19};
fracture_3 = news; Plane Surface(fracture_3) = {frac_loop_3};
Physical Surface("FRACTURE_3") = {fracture_3};
Surface{fracture_3} In Volume{1};

Line{frac_line_36} In Surface{fracture_3};
Line{frac_line_37} In Surface{fracture_3};
Line{frac_line_39} In Surface{fracture_3};

// End of fracture specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p19};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p20};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p21};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p22};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {p23};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {p24};
Physical Point("DOMAIN_BOUNDARY_POINT_6") = {p25};
Physical Point("DOMAIN_BOUNDARY_POINT_7") = {p26};
// End of physical point specification

