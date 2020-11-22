
// Define points
p0 = newp; Point(p0) = {-2.0, -2.0, 0.0, 0.3 };
p1 = newp; Point(p1) = {3.0, -2.0, 0.0, 0.3 };
p2 = newp; Point(p2) = {3.0, 3.0, 0.0, 0.3 };
p3 = newp; Point(p3) = {-2.0, 3.0, 0.0, 0.3 };
p4 = newp; Point(p4) = {0.0, 0.0, 0.0, 0.2 };
p5 = newp; Point(p5) = {2.0, 0.0, 0.0, 0.2 };
p6 = newp; Point(p6) = {1.0, 0.0, 0.0, 0.2 };
p7 = newp; Point(p7) = {1.0, 1.0, 0.0, 0.2 };
// End of point specification

// Start of specification of domain// Define lines that make up the domain boundary
bound_line_0 = newl;
Line(bound_line_0) ={p0, p1};
Physical Line("DOMAIN_BOUNDARY_0") = { bound_line_0 };
bound_line_1 = newl;
Line(bound_line_1) ={p1, p2};
Physical Line("DOMAIN_BOUNDARY_1") = { bound_line_1 };
bound_line_2 = newl;
Line(bound_line_2) ={p2, p3};
Physical Line("DOMAIN_BOUNDARY_2") = { bound_line_2 };
bound_line_3 = newl;
Line(bound_line_3) ={p3, p0};
Physical Line("DOMAIN_BOUNDARY_3") = { bound_line_3 };

// Line loop that makes the domain boundary
Domain_loop = newll;
Line Loop(Domain_loop) = {bound_line_0, bound_line_1, bound_line_2, bound_line_3};
domain_surf = news;
Plane Surface(domain_surf) = {Domain_loop};
Physical Surface("DOMAIN") = {domain_surf};
// End of domain specification

// Start specification of fractures/compartment boundary/auxiliary elements
frac_line_0 = newl; Line(frac_line_0) = {p4, p6};
Line{frac_line_0} In Surface{domain_surf};
frac_line_1 = newl; Line(frac_line_1) = {p5, p6};
Line{frac_line_1} In Surface{domain_surf};
Physical Line("FRACTURE_4") = { frac_line_0, frac_line_1 };

frac_line_2 = newl; Line(frac_line_2) = {p6, p7};
Line{frac_line_2} In Surface{domain_surf};
Physical Line("FRACTURE_5") = { frac_line_2 };

// End of /compartment boundary/auxiliary elements specification

// Start physical point specification
Physical Point("FRACTURE_POINT_0") = {p6};
// End of physical point specification

