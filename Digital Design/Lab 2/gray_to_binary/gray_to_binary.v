`timescale 1ns / 1ps
module gray_to_binary(g1,g2,g3,g4,a,b,c,d);
output a,b,c,d;
input g1,g2,g3,g4;
assign a = g1;
assign b = g1^g2;
assign c = g1^g2^g3;
assign d = g1^g2^g3^g4;
endmodule