`timescale 1ns / 1ps
module alu(a,b,op,result);

input [31:0] a,b;
input[2:0] op;
output reg [31:0] result;

always @(a,b,op,result)
 begin
  case(op)
   0 : result = 0;
   1 : result = a+b;
   2 : result = a-b;
   3 : result = a<<1;
   4 : result = a>>1;
   5 : result = a&b;
   6 : result = a|b;
   7 : result = a^b;
  endcase
 end

endmodule

