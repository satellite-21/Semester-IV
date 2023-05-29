`timescale 1ns / 1ps

module testBench;

reg [0:3]a;
reg [0:3]b;

wire greater,lesser,equal;
comparator com(a,b,greater,lesser,equal);
initial
 begin
 a = 4'b1100;b= 4'b1110;
 #20
 a = 4'b1010; b = 4'b0011;
 #20
 a = 4'b1101; b = 4'b1101 ;
 end
initial #80 $finish;

endmodule

