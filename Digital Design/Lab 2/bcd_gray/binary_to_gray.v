`timescale 1ns / 1ps
module binary_to_gray(a,b,c,d,g1,g2,g3,g4);
input a,b,c,d;
output g1,g2,g3,g4;
assign g1 = a;
assign g2 = a^b;
assign g3 = b^c;
assign g4 = c^d;
endmodule
