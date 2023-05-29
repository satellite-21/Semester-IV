`timescale 1ns / 1ps
module fourLine2(d,x,v);
input [3:0] d;
output reg [1:0] x;
output reg v;

always@(d)
 begin
 assign v = 1 ;
 casex(d)
  4'b1xxx : x = 2'b11;
  4'b01xx : x = 2'b10; 
  4'b001x : x = 2'b01;
  4'b0001 : x = 2'b00;
  default : 
   begin
    x = 2'bxx;
   assign v = 0;
   end
 endcase
 end
endmodule

