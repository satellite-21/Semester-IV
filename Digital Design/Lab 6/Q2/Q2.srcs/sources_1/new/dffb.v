`timescale 1ns / 1ps
// Code your design here
module dffb (q, d, clk, rst,q1);
 output q,q1;
 input d, clk, rst;
 reg q,q1;
 always @(posedge clk)
   if (rst) q = 1'b0;           //DFF using Blocking statements:
 else 
   begin
  q <= d;           //DFF using Non- Blocking statements:
  q1 <= q;          
   end
endmodule 
