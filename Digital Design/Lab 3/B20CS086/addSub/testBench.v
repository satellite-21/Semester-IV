`timescale 1ns / 1ps


module test_adder_subtractor_unit;

reg a0,a1,a2,a3,b0,b1,b2,b3,m;
wire s0,s1,s2,s3,v;

adderSubtractor as(a0,a1,a2,a3,b0,b1,b2,b3,m,s0,s1,s2,s3,v);

initial
 begin
 a0 = 1'b1; a1 = 1'b1; a2 = 1'b0; a3 = 1'b0; 
 b0 = 1'b1; b1 = 1'b1; b2 = 1'b1; b3 = 1'b0; 
 m = 1'b0; 
 #20
 m = 1'b1;
 #20 
 a0 = 1'b0; a1 = 1'b1; a2 = 1'b1; a3 = 1'b0; 
 b0 = 1'b1; b1 = 1'b0; b2 = 1'b0; b3 = 1'b0; 
 m = 1'b0; 
 #20
 m = 1'b1; 
 end
initial #80 $finish;
endmodule

